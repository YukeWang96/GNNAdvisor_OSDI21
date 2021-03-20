#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define MAX_DIM 128
#define MAX_NB 32           // <= partsize 

#define threadPerWarp 32    //must < 32
#define wrapPerBlock 8      

__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts
);

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts
);


////////////////////////////////////////////
//
// Foward Pass (GCN)  node update --> neighbor aggregation
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int threadPerBlock
) 
{
    auto tmp = torch::mm(input, weight);
    auto output = torch::zeros_like(tmp);

    const int dim = tmp.size(1);
    const int num_nodes = tmp.size(0);
    const int num_parts = part2Node.size(0);

    const int block_size = wrapPerBlock * threadPerWarp;
    const int blocks = (num_parts * 32 + block_size  - 1) / block_size; 

    printf("grid: %d, block: %d\n", blocks, block_size);
    printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    printf("input: (%d, %d)", tmp.size(0), tmp.size(1));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward", ([&] {
                                spmm_forward_cuda_kernel<scalar_t><<<blocks, block_size>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts
                                );
                            }));
                                 
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output};
}

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / 32;                             // global warp-id
    int block_warpId = threadIdx.x / 32;               // block warp-id
    int laneid = threadIdx.x % 32;                     // warp thread-id -- laneid

    if (warpId < num_parts && laneid < threadPerWarp){

        // if (laneid == 0)
        //     printf("%d\n", warpId);

        __shared__  int partial_ids[MAX_NB * wrapPerBlock];
        __shared__ float partial_results[MAX_DIM * wrapPerBlock];

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end
        float src_norm = degrees[srcId];            // norm of the source node

        // Cache the part neighbors.
        const int pindex_base = block_warpId * MAX_NB;
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += threadPerWarp){
            partial_ids[pindex_base + laneid] = column_index[nidx];
        }

        //  __syncthreads();
         __syncwarp();

        // intra-part aggregation of all neighbors
        const int presult_base = block_warpId * MAX_DIM;
        // printf("1--block_warpId: %d, MAX_DIM: %d, presult_base: %d\n", block_warpId, MAX_DIM, presult_base);

        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                #pragma unroll
                for (int d = laneid; d < dim; d += threadPerWarp){
                    partial_results[block_warpId * MAX_DIM + d] = 0;
                }
            
            #pragma unroll
            for (int d = laneid; d < dim; d += threadPerWarp){
                // printf("2--block_warpId: %d, MAX_DIM: %d, presult_base: %d\n", block_warpId, MAX_DIM, presult_base);
                // printf("3--presult_base: %d\n", presult_base);
                // printf("nid: %d, d: %d, input[nid][d]: %d, presult_base: %d\n", nid, d, input[nid][d], presult_base);
                // partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nIndex][d], 0);
                partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        #pragma unroll
        for (int d = laneid; d < dim; d += threadPerWarp){
            atomicAdd_F((float*)&output[srcId][d], partial_results[presult_base + d]);
        }
    }
}

////////////////////////////////////////////
// 
// backward pass (GCN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_backward_cuda(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int threadPerBlock
) {

    auto d_input_prime = torch::zeros_like(d_output);

    const int dim = d_input_prime.size(1);
    const int num_nodes = d_input_prime.size(0);
    const int num_parts = part2Node.size(0);

    const int block_size = wrapPerBlock * threadPerWarp;
    const int blocks = (num_parts * 32 + block_size - 1) / block_size; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward", ([&] {
                                spmm_backward_cuda_kernel<scalar_t><<<blocks, block_size>>>(
                                    d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts
                                );
                            }));
    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    auto d_input = torch::mm(d_input_prime, W.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);

    return {d_input, d_weight};
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int warpId =  tid / 32;
    int laneid = tid % 32;
    int block_warpId = threadIdx.x/32;
    
    if (warpId < num_parts && laneid < threadPerWarp){

        __shared__  int partial_index[MAX_NB * wrapPerBlock];
        __shared__ float partial_results[MAX_DIM * wrapPerBlock];

        int srcId = part2Node[warpId];
        int partBeg = part_pointers[warpId];
        int partEnd = part_pointers[warpId + 1];
        float src_norm = degrees[srcId];

        int pindex_base = block_warpId * MAX_NB;
        for (int nid = partBeg + laneid; nid < partEnd; nid += threadPerWarp){
            partial_index[pindex_base + nid - partBeg] = column_index[nid];
        }
         __syncthreads();

        int presult_base = block_warpId * MAX_DIM;
        for (int nid = 0; nid < partEnd - partBeg; nid++)
        {
            int nIndex = partial_index[pindex_base + nid];
            float degree_norm =  __fmaf_rn(src_norm, degrees[nIndex], 0);

            if (nid == 0)
                #pragma unroll
                for (int d = laneid; d < dim; d += threadPerWarp){
                    partial_results[presult_base + d] = 0;
                }
            
            #pragma unroll
            for (int d = laneid; d < dim; d += threadPerWarp){
                partial_results[presult_base + d] += __fmaf_rn(degree_norm, d_output[nIndex][d], 0);
            }
        }
        for (int d = laneid; d < dim; d += threadPerWarp){
            atomicAdd_F((float*)&d_input[srcId][d], partial_results[presult_base + d]);
        }
    }
}

////////////////////////////////////////////
//
// Foward Pass (GIN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_forward_cuda_gin(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int threadPerBlock
) 
{
    auto tmp = torch::zeros_like(input);
    const int dim = tmp.size(1);
    const int num_nodes = tmp.size(0);
    const int num_parts = part2Node.size(0);

    const int block_size = wrapPerBlock * threadPerWarp;
    const int blocks = (num_parts * 32 + block_size  - 1) / block_size; 

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward_gin", ([&] {
                                spmm_forward_cuda_kernel<scalar_t><<<blocks, threadPerBlock>>>(
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts
                                );
                            }));
    
    auto output = torch::mm(tmp, weight);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return {output, tmp};
}

////////////////////////////////////////////
// 
// backward pass (GIN)
//
////////////////////////////////////////////
std::vector<torch::Tensor> spmm_backward_cuda_gin(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int threadPerBlock
) {

    auto d_weight = torch::mm(X.transpose(0,1), d_output);
    auto d_input_prime = torch::mm(d_output, W.transpose(0,1));
    auto d_input = torch::zeros_like(d_input_prime);

    const int dim = d_input.size(1);
    const int num_nodes = d_input.size(0);
    const int num_parts = part2Node.size(0);

    const int block_size = wrapPerBlock * threadPerWarp;
    const int blocks = (num_parts * 32 + block_size - 1) / block_size; 

    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward_gin", ([&] {
                                spmm_backward_cuda_kernel<scalar_t><<<blocks, block_size>>>(
                                    d_input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts
                                );
                            }));

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {d_input, d_weight};
}
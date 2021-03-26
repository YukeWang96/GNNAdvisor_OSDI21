#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define threadPerWarp 32
#define smem_gap 250

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
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);

template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel_gin(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    float epsilon,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
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
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel_gin(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    float epsilon,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) 
{
    auto tmp = torch::mm(input, weight);
    auto output = torch::zeros_like(tmp);
    // printf("input: (%d, %d)\n", input.size(0), input.size(1));
    // printf("weight: (%d, %d)\n", weight.size(0), weight.size(1));
    // auto output = torch::zeros((input.size(0), weight.size(1)));

    const int dim = output.size(1);
    const int num_nodes = output.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * threadPerWarp;
    const int grid = (num_parts * 32 + block  - 1) / block; 
    int shared_memory = warpPerBlock * dim * sizeof(float);

    // printf("grid: %d, block: %d\n", grid, block);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
    // printf("dimWorker: %d\n", dimWorker);
    // printf("shared_memory: %d\n", tmp.size(0), tmp.size(1));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward", ([&] {
                                spmm_forward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize,
                                    dimWorker,
                                    warpPerBlock
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
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / 32;                             // global warp-id
    int block_warpId = threadIdx.x / 32;               // block warp-id
    int laneid = threadIdx.x % 32;                     // warp thread-id -- laneid

    extern __shared__ float part_info[];                    // part information.
    // int *partial_ids = (int*) part_info;                    // caching ids
    // int part_shift_base = partSize * warpPerBlock;    // shared memory shift for partial result.
    float *partial_results = part_info; // + part_shift_base;   // caching partial results.
    int partial_ids[256];
    
    if (warpId < num_parts){

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end
        float src_norm = degrees[srcId];            // norm of the source node

        // Cache the part neighbors by all threads from a warp.
        // const int pindex_base = block_warpId * partSize;
        // #pragma unroll
        // for (int nidx = partBeg + laneid; nidx < partEnd; nidx += threadPerWarp){
        //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
        //     partial_ids[pindex_base + laneid] = column_index[nidx];
        // }
        
        #pragma unroll
        for (int nidx = partBeg; nidx < partEnd; nidx++){
        //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[nidx - partBeg] = column_index[nidx];
        }
        
        __syncwarp();

        // if (laneid == 0)
        // for (int nIdx = laneid; nIdx < partEnd - partBeg; nIdx++){
            // int nid = partial_ids[pindex_base + nIdx];
            // int nid = partial_ids[nIdx];
            // printf("verify nid - 111111: %d\n", nid);
            // if(nid >= num_nodes || nid < 0) printf("verify nid: %d\n", nid);
        // }

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            // int nid = partial_ids[pindex_base + nIdx];
            int nid = partial_ids[nIdx];
            // if (laneid == 0)
            //     printf("verify nid - 222222: %d\n", nid);
            float degree_norm_inv = __fmaf_rn(src_norm, degrees[nid], 0);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                // printf("nid: %d, dim: %d, input[nid][d]: %d\n", nid, d, input[nid][d]);
                // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                partial_results[presult_base + d] += __fmaf_rn(degree_norm_inv, input[nid][d], 0);
                // partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
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
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) {

    auto d_input_prime = torch::zeros_like(d_output);

    const int dim = d_input_prime.size(1);
    const int num_nodes = d_input_prime.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * threadPerWarp;
    const int grid = (num_parts * 32 + block - 1) / block; 
    // const int shared_memory = warpPerBlock * partSize * sizeof(int) + warpPerBlock * dim * sizeof(float);
    const int shared_memory = warpPerBlock * dim * sizeof(float);

    // AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward", ([&] {
    //                             spmm_backward_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
    //                                 d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
    //                                 d_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
    //                                 row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    //                                 column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    //                                 degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    //                                 part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
    //                                 part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
    //                                 num_nodes, 
    //                                 dim,
    //                                 num_parts,
    //                                 partSize,
    //                                 dimWorker,
    //                                 warpPerBlock
    //                             );
    //                         }));
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
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int warpId =  tid / threadPerWarp;
    int block_warpId = threadIdx.x / threadPerWarp;
    int laneid = threadIdx.x % threadPerWarp;

    extern __shared__ float part_info[];                        // part information.
    // int *partial_ids = (int*) part_info;                    // caching ids
    // const int part_shift_base = partSize * warpPerBlock;    // shared memory shift for partial result.
    // float *partial_results = (float*)&part_info[part_shift_base];   // caching partial results.
    float *partial_results = part_info;                         // caching partial results.
    int partial_ids[256];

    if (warpId < num_parts){

        int srcId = part2Node[warpId];
        int partBeg = part_pointers[warpId];
        int partEnd = part_pointers[warpId + 1];
        float src_norm = degrees[srcId];

        // const int pindex_base = block_warpId * partSize;
        // #pragma unroll
        // for (int nid = partBeg + laneid; nid < partEnd; nid += threadPerWarp){
            // partial_ids[pindex_base + laneid] = column_index[nid];
        // }
        
        #pragma unroll
        for (int nidx = partBeg; nidx < partEnd; nidx++){
        //     if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[nidx - partBeg] = column_index[nidx];
        }

        __syncwarp();

        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            // int nid = partial_ids[pindex_base + nIdx];
            int nid = partial_ids[nIdx];

            float degree_norm =  __fmaf_rn(src_norm, degrees[nid], 0);

            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += __fmaf_rn(degree_norm, d_output[nid][d], 0);
            }
        }

        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
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
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) 
{
    auto tmp = torch::zeros_like(input);
    const int dim = tmp.size(1);
    const int num_nodes = tmp.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * threadPerWarp;
    const int grid = (num_parts * 32 + block  - 1) / block; 
    const int shared_memory = warpPerBlock * partSize * sizeof(int) + warpPerBlock * dim * sizeof(float) + smem_gap * sizeof(float);

    // printf("grid: %d, block: %d\n", grid, block);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("input: (%d, %d)\n", tmp.size(0), tmp.size(1));
    // printf("dimWorker: %d\n", dimWorker);
    // printf("warpPerBlock: %d, shared_memory: %d\n", warpPerBlock, shared_memory);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "spmm_cuda_forward_gin", ([&] {
                            spmm_forward_cuda_kernel_gin<scalar_t><<<grid, block, shared_memory>>>(
                                    tmp.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    epsilon,
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize, 
                                    dimWorker, 
                                    warpPerBlock
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


template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel_gin(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    float epsilon,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;  // global thread-id
    int warpId = tid / 32;                             // global warp-id
    int block_warpId = threadIdx.x / 32;               // block warp-id
    int laneid = threadIdx.x % 32;                     // warp thread-id -- laneid

    extern __shared__ float part_info[];                                // part information.
    const int part_shift_base = partSize * warpPerBlock;                // shared memory shift for partial result.
    int *partial_ids = (int*) part_info;                                // caching ids
    float *partial_results = part_info + part_shift_base + smem_gap;   // caching partial results.

    if (warpId < num_parts){

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end

        // Cache the part neighbors.
        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += dimWorker){
            partial_ids[pindex_base + laneid] = column_index[nidx];
        }

         __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&output[srcId][d], epsilon*partial_results[presult_base + d]);
        }
    }
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
    float epsilon,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
) {

    auto d_weight = torch::mm(X.transpose(0,1), d_output);
    auto d_input_prime = torch::mm(d_output, W.transpose(0,1));
    auto d_input = torch::zeros_like(d_input_prime);

    const int dim = d_input.size(1);
    const int num_nodes = d_input.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * threadPerWarp;
    const int grid = (num_parts * 32 + block - 1) / block; 
    const int shared_memory = warpPerBlock * partSize * sizeof(int) + warpPerBlock * dim * sizeof(float) + smem_gap * sizeof(float);

    // printf("warpPerBlock: %d, shared_memory: %d\n", warpPerBlock, shared_memory);
    AT_DISPATCH_FLOATING_TYPES(d_output.type(), "spmm_cuda_backward_gin", ([&] {
                                spmm_backward_cuda_kernel_gin<scalar_t><<<grid, block, shared_memory>>>(
                                    d_input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    d_input_prime.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    epsilon,
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize, 
                                    dimWorker, 
                                    warpPerBlock
                                );
                            }));

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {d_input, d_weight};
}

template <typename scalar_t>
__global__ void spmm_backward_cuda_kernel_gin(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_input,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_output,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    float epsilon,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;
    int warpId =  tid / 32;
    int block_warpId = threadIdx.x / 32;
    int laneid = threadIdx.x % 32;

    extern __shared__ float part_info[];                                // part information.
    const int part_shift_base = partSize * warpPerBlock;                // shared memory shift for partial result.
    int *partial_ids = (int*) part_info;                                // caching ids
    float *partial_results = part_info + part_shift_base + smem_gap;   // caching partial results.
    
    if (warpId < num_parts){

        int srcId = part2Node[warpId];
        int partBeg = part_pointers[warpId];
        int partEnd = part_pointers[warpId + 1];

        const int pindex_base = block_warpId * partSize;
        #pragma unroll
        for (int nid = partBeg + laneid; nid < partEnd; nid += dimWorker){
            partial_ids[pindex_base + laneid] = column_index[nid];
        }

        __syncwarp();

        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];

            if (nIdx == 0)
                #pragma unroll
                if (laneid < dimWorker)
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += d_output[nid][d];
            }
        }

        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&d_input[srcId][d], epsilon*partial_results[presult_base + d]);
        }
    }
}
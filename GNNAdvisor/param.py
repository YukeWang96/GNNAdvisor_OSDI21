import math 

# package of input parameters
class inputProperty(object):
    def __init__(self, row_pointers=None, column_index=None, 
                degrees=None, partPtr=None, part2Node=None, 
                partSize=None, dimWorker=None, warpPerBlock=None, 
                sharedMem=None,
                hiddenDim=None,
                dataset_obj=None,
                enable_rabbit=False,
                manual_mode=True,
                verbose=False):
        
        if dataset_obj is None:
            raise ValueError("Dataset object MUST SET !!!")

        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.partPtr = partPtr 
        self.part2Node = part2Node 
        self.verbose_flag = verbose

        self.partSize = partSize
        self.dimWorker = dimWorker
        self.warpPerBlock = warpPerBlock

        self.dimWorker_input = dimWorker
        self.dimWorker_hidden = dimWorker
        self.warpPerBlock_input = warpPerBlock
        self.warpPerBlock_hidden = warpPerBlock

        self.num_nodes = dataset_obj.num_nodes
        self.avgNodeDegree = dataset_obj.avg_degree
        self.avgEdgeSpan = dataset_obj.avg_edgeSpan
        self.inputDim = dataset_obj.num_features
        self.hiddenDim = hiddenDim

        self.manual_mode = manual_mode
        self.state_set_input = False
        self.enable_rabbit = enable_rabbit
        self.reorder_status = False
        
        self.dataset_obj = dataset_obj
        self.MAX_warpPerBlock = 8               # for better scheduling.
        self.share_memory = sharedMem * 0.4         
        self.gap_smem = 100

    def decider(self):
        '''
        Determine the performance-related parameter here.
        manual_mode: using user-specified parameters
        auto_mode:   determining the parameters according to the GPU resources and scheduling performance consideration.
        '''

        if self.manual_mode:
            if self.enable_rabbit:
                self.dataset_obj.reorder_flag = True
                self.dataset_obj.rabbit_reorder()
                self.reorder_status = True
            else:
                self.dataset_obj.reorder_flag = False
                self.reorder_status = False

            if self.verbose_flag:
                print("\n=> MANUAL Config Complete !!!\n")
        else:
            # Determine the neighbor partitioning.
            self.partSize = int(self.avgNodeDegree)

            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.inputDim * 4 + self.gap_smem * 4)/1e3
            if self.verbose_flag:
                print("input-layer shared memory (KB): {:.3f} ".format(est_shared))
            share_memory_input = min(est_shared, self.share_memory)
            if self.verbose_flag:
                print("input-layer updated (KB): {:.3f}".format(share_memory_input))

            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.hiddenDim + 4 * self.gap_smem)/1e3
            if self.verbose_flag:
                print("hidden-layer shared memory (KB): {:.3f}".format(est_shared))
            share_memory_hidden = min(est_shared, self.share_memory)
            if self.verbose_flag:
                print("hidden-layer updated (KB): {:.3f}".format(share_memory_hidden))

            # Determine the warpPerBlock for input and hidden layer.
            self.warpPerBlock_input = int(share_memory_input * 1e3 / (self.partSize * 4 + self.inputDim * 4))
            self.warpPerBlock_hidden = int(share_memory_hidden * 1e3 / (self.partSize * 4 + self.hiddenDim * 4))
            
            self.warpPerBlock_input = min(self.warpPerBlock_input, self.MAX_warpPerBlock)
            self.warpPerBlock_hidden = min(self.warpPerBlock_hidden, self.MAX_warpPerBlock)

            # Determine the dimWorker_input for input layer.
            if self.inputDim > 32:
                self.dimWorker_input = 32
            else:
                self.dimWorker_input = self.inputDim
            
            # Determine the dimWorker_hidden for hidden layer.
            if self.hiddenDim > 32:
                self.dimWorker_hidden = 32
            else:
                self.dimWorker_hidden = self.hiddenDim

            if self.enable_rabbit:
                # Determine whether to reorder a graph.
                if math.sqrt(self.avgEdgeSpan) > math.sqrt(self.num_nodes)/100:
                    self.dataset_obj.reorder_flag = True
                    self.reorder_status = True
                else:
                    self.dataset_obj.reorder_flag = False
                    self.reorder_status = False
                
                self.dataset_obj.rabbit_reorder()

            if self.verbose_flag:
                print("\n=> AUTO Decider Complete !!!\n")

    def set_input(self):
        '''
        Determine the performance-related parameter for input layer.
        Switch the parameter for input layer.
        '''
        self.dimWorker = self.dimWorker_input        
        self.warpPerBlock = self.warpPerBlock_input
        self.state_set_input = True

        return self        
    
    def set_hidden(self):
        '''
        Determine the performance-related parameter for hidden layer.
        Switch the parameter for hidden layer.
        '''
        self.dimWorker = self.dimWorker_hidden        
        self.warpPerBlock = self.warpPerBlock_hidden
        self.state_set_input = False
        return self   

    def print_param(self):
        if self.verbose_flag:
            if self.state_set_input:
                if self.manual_mode:
                    print("# manual INPUT partSize: {}".format(self.partSize))
                    print("# manual INPUT dimWorker: {}".format(self.dimWorker))
                    print("# manual INPUT warpPerBlock: {}".format(self.warpPerBlock))
                else:
                    print("# auto INPUT partSize: {}".format(self.partSize))
                    print("# auto INPUT dimWorker: {}".format(self.dimWorker))
                    print("# auto INPUT warpPerBlock: {}".format(self.warpPerBlock))
                    print("# auto INPUT reorder_flag: {}".format(self.reorder_status))
            else:
                if self.manual_mode:
                    print("# manual HIDDEN partSize: {}".format(self.partSize))
                    print("# manual HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# manual HIDDEN warpPerBlock: {}".format(self.warpPerBlock))
                else:
                    print("# auto HIDDEN partSize: {}".format(self.partSize))
                    print("# auto HIDDEN dimWorker: {}".format(self.dimWorker))
                    print("# auto HIDDEN warpPerBlock: {}".format(self.warpPerBlock))
                    print("# auto HIDDEN reorder_flag: {}".format(self.reorder_status))
# package of input parameters
class inputProperty(object):
    def __init__(self, row_pointers=None, column_index=None, 
                degrees=None, partPtr=None, 
                part2Node=None, partSize=None, dimWorker=None, warpPerBlock=None,
                avgNodeDegree=None, 
                inputDim=None,
                hiddenDim=None,
                manual_mode=True):
                
        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.partPtr = partPtr 
        self.part2Node = part2Node 

        self.partSize = partSize
        self.dimWorker = dimWorker
        self.warpPerBlock = warpPerBlock

        self.avgNodeDegree = avgNodeDegree
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim

        self.manual_mode = manual_mode

        self.MAX_warpPerBlock = 8
        self.share_memory = 96 # 96 KB for RTX3090

    def decider(self):
        '''
        Determine the performance-related parameter here.
        (True): manual_mode 
        (False): auto_mode
        '''
        if self.manual_mode:
            print("MANUAL Complete !!!")
            return
        else:
            # Determine the neighbor partitioning.
            self.partSize = int(self.avgNodeDegree)

            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.inputDim * 4)/1e3
            print("input shared: {} KB".format(est_shared))
            share_memory_input = min(est_shared, self.share_memory)
            
            est_shared = self.MAX_warpPerBlock * (self.partSize * 4 + self.hiddenDim * 4)/1e3
            print("hidden shared: {} KB".format(est_shared))
            share_memory_hidden = min(est_shared, self.share_memory)

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

            print()
            print("** AUTO Decider Complete !!!")

    
    def set_input(self):
        '''
        Determine the performance-related parameter for input layer.
        '''
        self.dimWorker = self.dimWorker_input        
        self.warpPerBlock = self.warpPerBlock_input

        return self        
    
    def set_hidden(self):
        '''
        Determine the performance-related parameter for hidden layer.
        '''
        self.dimWorker = self.dimWorker_hidden        
        self.warpPerBlock = self.warpPerBlock_hidden

        return self   

    def print_param(self):
        if not self.manual_mode:
            print("# auto partSize: {}".format(self.partSize))
            print("# auto dimWorker: {}".format(self.dimWorker))
            print("# auto warpPerBlock: {}".format(self.warpPerBlock))
        else:
           print("# manual partSize: {}".format(self.partSize))
           print("# manual dimWorker: {}".format(self.dimWorker))
           print("# manual warpPerBlock: {}".format(self.warpPerBlock))

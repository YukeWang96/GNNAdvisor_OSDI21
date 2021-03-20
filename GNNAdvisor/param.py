# package of input parameters
class inputProperty(object):
    def __init__(self, row_pointers=None, column_index=None, 
                degrees=None, partPtr=None, 
                part2Node=None, partSize=None, dimWorker=None, warpPerBlock=None):
                
        self.row_pointers = row_pointers
        self.column_index = column_index
        self.degrees = degrees
        self.partPtr = partPtr 
        self.part2Node = part2Node 
        self.partSize = partSize
        self.dimWorker = dimWorker
        self.warpPerBlock = warpPerBlock
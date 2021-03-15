import GNNAdvisor as GNNA
import torch
# import other packages ...

# Create a GCN class.
class GCN(torch.nn.Module):
    def __init__(self, inDim, hiDim, outDim, nLayers):
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNNA.GCNConv(inDim, hiDim))
        for i in range(nLayers - 2):
            layer = GNNA.GCNConv(hiDim, hiDim)
            self.layers.append(layer)
        self.layers.append(GNNA.GCNConv(hiDim, outDim))
        self.softmax = torch.nn.Softmax()
        
    def forward(self, X, graph, param):
        for i in range(len(self.layers)):
            X = self.layers[i](X, graph, param)
            X = self.ReLU(X)
        X = self.softmax(X)
        return X

# Define a two-layer GCN model.
model = GCN(inDim=100, hiDim=16, outDim=10, nLayers=2)

# Loading graph and extracting input propertities.
graphObj, inputInfo = GNNA.LoaderExtractor(graphFile, 
                                            model)
# Set runtime parameters automatically.
X, graph, param = GNNA.Decider(graphObj, inputInfo)

# Run model.
predict_y = model(X, graph, param)

# Compute loss and accuracy.
# Gradient backpropagation for training.
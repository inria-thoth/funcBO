import torch
import torch.nn as nn

class NeuralNetworkOuterModel(nn.Module):
  """
  A neural network for the outer model parametrized by mu.
  """
  def __init__(self, layer_sizes):
    super(NeuralNetworkOuterModel, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1], bias=False)
    #nn.init.normal_(self.layer_1.weight)
    nn.init.ones_(self.layer_1.weight)

  def forward(self, x):
    x = self.layer_1(x)
    return x
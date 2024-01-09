import torch
import torch.nn as nn

class NeuralNetworkInnerModel(nn.Module):
  """
  A neural network to approximate the function h* for Neur. Imp. Diff.
  """
  def __init__(self, layer_sizes):
    self.output_dim = layer_sizes[-1]
    super(NeuralNetworkInnerModel, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)
    self.layer_2 = nn.Linear(layer_sizes[1], layer_sizes[2])
    nn.init.kaiming_uniform_(self.layer_2.weight)
    self.layer_3 = nn.Linear(layer_sizes[2], layer_sizes[3])
    nn.init.kaiming_uniform_(self.layer_3.weight)
    self.layer_4 = nn.Linear(layer_sizes[3], layer_sizes[4])

  def forward(self, x):
    res1 = torch.relu(self.layer_1(x))
    res2 = torch.tanh(self.layer_2(res1))
    res3 = torch.tanh(self.layer_3(res2))
    res = self.layer_4(res3)
    return res
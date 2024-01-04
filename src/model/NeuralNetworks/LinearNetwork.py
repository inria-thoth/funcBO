import torch
import torch.nn as nn

class LinearNetwork(nn.Module):
  """
  A neural network
  """
  def __init__(self, layer_sizes):
    self.output_dim = layer_sizes[-1]
    super(LinearNetwork, self).__init__()
    self.layer_1 = nn.Linear(layer_sizes[0], layer_sizes[1])
    nn.init.kaiming_uniform_(self.layer_1.weight)

  def forward(self, x):
    res = self.layer_1(x)
    main = res[:, 0:1]
    aux = torch.hstack((main, main, main, main))
    return (main, aux)
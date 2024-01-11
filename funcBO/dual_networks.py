import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict


class DualNetwork(nn.Module):
  def __init__(self,network, make_deepcopy=True):
    super(DualNetwork,self).__init__()
    if make_deepcopy:
      self.dual_network = deepcopy(network)
    else:
      self.dual_network = network

  def forward(self,inputs):
    return  self.dual_network(inputs)


class LinearDualNetwork(DualNetwork):
  """
  Outer layer is the layer that gives the features phi(Z),
  by default none, then just takes the layer before the last.
  """
  def __init__(self,network, 
                    network_inputs, 
                    make_deepcopy = False,
                    output_layer = None):
    super(LinearDualNetwork,self).__init__(network,make_deepcopy=make_deepcopy)
    if not output_layer:
      # List of all layers assuming that the network is sequential
      L = len(network._modules.keys())
      # At least two layers to have the one before the last
      # TODO: this does not work when the network is a Sequential object, it then considers that there is only one layer
      assert L>=2
      for i, name in enumerate(network._modules.keys()):
        if i==L-2:
      #for name, param in network.named_parameters():
          output_layer = name
    dummpy_param = next(network.parameters())
    device= dummpy_param.device
    dtype = dummpy_param.dtype

    self.model_with_hook = ModelWithHook(output_layer,network)
    
    out, selected_out = self.model_with_hook(network_inputs)
    self.out_shape = out.shape[1:]
    in_dim = selected_out.flatten(start_dim=1).shape[1]
    out_dim = out.flatten(start_dim=1).shape[1]

    self.linear = torch.nn.Linear(in_dim+1, out_dim,bias=False, 
                                device=device, 
                                dtype=dtype)

  def forward(self, inputs,with_features=False):
    with torch.no_grad():
      out, selected_out = self.model_with_hook(inputs)
      selected_out = selected_out.detach().flatten(start_dim=1)
      ones = torch.ones(selected_out.shape[0],1, 
                        dtype=selected_out.dtype,
                        device=selected_out.device)
      selected_out = torch.cat((selected_out,ones),dim=1)
    out = self.linear(selected_out)
    out = torch.unflatten(out,dim=1,sizes=self.out_shape)
    if with_features:
      return out, selected_out 
    else:
      return out

  def parameters(self):
      return (self.linear.parameters()) 


class ModelWithHook(nn.Module):
    def __init__(self, output_layers, model):
        super(ModelWithHook,self).__init__()
        self.output_layers = output_layers
        #print(self.output_layers)
        self.selected_out = None
        #PRETRAINED MODEL
        self.model = model
        self.fhooks = []

        for i,l in enumerate(list(self.model._modules.keys())):
            if l==self.output_layers:
                self.fhooks.append(getattr(self.model,l).register_forward_hook(self.forward_hook(l)))
    
    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out = output
        return hook

    def forward(self, x):
        out = self.model(x)
        return out, self.selected_out

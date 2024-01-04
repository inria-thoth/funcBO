import torch
from torch.autograd import Variable
from torch.autograd.functional import hvp
import torchviz
from torchviz import make_dot

# Define a simple twice-differentiable function
def f(x):
    return x.sin().sum()

# Input values
x = Variable(torch.randn(3, 1), requires_grad=False)
v = Variable((torch.Tensor([[1, 2, 3]])).T, requires_grad=True)

# Compute hessian vector product
hessvp = hvp(f, x, v)[1]

# Define the loss
loss = (hessvp)

# Create a graph of the loss
make_dot(loss, params={"v":v, "x":x, "hessvp":hessvp}, show_attrs=True, show_saved=True).render("hvp_graph_loss", format="png")
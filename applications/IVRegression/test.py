import torch 
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
        
    def forward(self, x):
        for _ in range(10):
            x = self.fc1(x)
        return x

class NewModel(nn.Module):
    def __init__(self,model):
        super(NewModel,self).__init__()
        self.model = model
        self.linear = torch.nn.Linear(1, 1,bias=False)
    def forward(self,x):
        with torch.no_grad():
            x = self.model(x).detach()
        return self.linear(x)
    def parameters(self):
        return (self.linear.parameters())

model = MyModel()
new_model = NewModel(model)
x = torch.randn(1, 1)


params = list(model.parameters())

print('model params:'+ str(params))





params = list(new_model.parameters())

print('new model params:'+ str(params))






out = new_model(x)
out.backward()


params = list(new_model.parameters())
params[0].grad





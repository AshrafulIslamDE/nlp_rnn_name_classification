import torch
import torch.nn as nn
x=torch.ones(3,3)
print(x)
zeros=torch.zeros(3,x.size(1))
x=torch.cat((x,zeros),dim=1)
print(x)
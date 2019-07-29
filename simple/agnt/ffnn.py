import torch 
import torch.nn as nn

class net(nn.Module):
  def __init__(self):
    super(net, self).__init__()

    self.l1 = nn.Linear(2, 20)
    self.r1 = nn.LeakyReLU()
    self.l2 = nn.Linear(20,2)

  def forward(self, x):
    x = self.r1(self.l1(x))
    x = self.l2(x)

    return x
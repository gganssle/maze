import numpy as np 
import torch 
import torch.nn as nn
from agnt import ffnn
from env import simple, reward

model = ffnn.net()
world = simple.env()
rwrd = reward.score()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(1000):
  state = world.get_state()

  q_vals = model(torch.Tensor(state))
  action = np.argmax(q_vals.data.numpy())

  target = torch.Tensor(state)

  loss = criterion(q_vals, target)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

torch.save(model.state_dict(), 'temp')

import numpy as np 
import torch 
import torch.nn as nn
from agnt import ffnn
from env import simple, reward

model = ffnn.net()
world = simple.env()
rwrd = reward.score()

model.load_state_dict(torch.load('temp'))
model.eval()

score_hist = []

for i in range(1000):
  state = world.get_state()

  #action = np.random.choice([0,1])
  q_vals = model(torch.Tensor(state)).data.numpy()
  action = np.argmax(q_vals)

  score_hist.append(rwrd.vanilla(action, state))

score_hist = np.array(score_hist)
print('accuracy:', 100 * score_hist.sum() / score_hist.shape[0], '%')

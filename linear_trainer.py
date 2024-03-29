import numpy as np 
import torch
import torch.nn as nn
from agnt import linear_ffnn
from env import linear, reward, discount
from tqdm import trange

actor = linear_ffnn.agent(train=True)
actor.model.train()

egreedy = False
eperc = [.0001, .9999]

optionswords = ['left', 'right', 'sit']
plot = True
num_games = 100000
max_iter = 20
discount_factor = 0.9
learning_rate = 0.01
model_checkpoint = f'/Users/gram/maze/dat/checkpoints/temp'
loss_history = np.array([])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(actor.model.parameters(), lr=learning_rate)

rwrd = reward.score()
dcount = discount.disc()

for game in trange(num_games):
  env = linear.world()        

  state = env.initial_state
  cursor = env.cursor
  local_reward = rwrd.initial_reward

  for i in range(max_iter):
    # for time step 0
    q_vals = actor.decision(state, cursor, env.end, [local_reward])
    if egreedy:
      action = np.random.choice(
          [np.argmax(q_vals.data.numpy()),
          np.random.randint(0,3)
          ], 
          p=eperc
        )
    else:
      action = np.argmax(q_vals.data.numpy())

    # for time step 1
    state1, cursor1 = env.get_state(state.copy(), cursor.copy(), optionswords[action])
    #local_reward1 = rwrd.distance(np.array(cursor1), env.end)
    local_reward = rwrd.dontstandstill(cursor1, env.end)
    #local_reward1 = rwrd.endonly(cursor1, env.end)
    #local_reward1 = rwrd.positivemoves(cursor1, env.end)

    q_vals1 = actor.decision(state1, cursor1, env.end, [local_reward])
    if egreedy:
      action1 = np.random.choice(
          [np.argmax(q_vals1.data.numpy()),
          np.random.randint(0,3)
          ], 
          p=eperc
        )
    else:
      action1 = np.argmax(q_vals1.data.numpy())

    # calculate target
    targetele = local_reward + discount_factor * q_vals1[action1]
    target = q_vals.clone()
    target[action] = targetele

    # backpropagate errors
    #print(cursor, env.end, local_reward, optionswords[action1])
    #print(q_vals)
    #print(q_vals1)
    #print(target)
    loss = criterion(q_vals, target)
    loss_history = np.append(loss_history, loss.data.numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # step the game
    state = state1
    cursor = cursor1

  if game % 1000 == 0:
    torch.save(actor.model.state_dict(), model_checkpoint)
    np.save('dat/loss_hist.npy', loss_history)

torch.save(actor.model.state_dict(), model_checkpoint)

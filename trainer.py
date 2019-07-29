import numpy as np 
import torch
import torch.nn as nn
from agnt import ffnn 
from env import lumpy, reward, discount
from tqdm import trange

actor = ffnn.agent(train=True)
actor.model.train()

egreedy = False
eperc = [.0001, .9999]

optionswords = ['left', 'right', 'up', 'down']
plot = True
num_games = 10
max_iter = 100
discount_factor = 0.9
learning_rate = 0.1
model_checkpoint = f'/Users/gram/maze/dat/checkpoints/temp'
loss_history = np.array([])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(actor.model.parameters(), lr=learning_rate)

rwrd = reward.score()
dcount = discount.disc()

for game in trange(num_games):
  env = lumpy.world(gridsize=20)        

  state = env.initial_state
  cursor = env.cursor
  local_reward = rwrd.initial_reward

  for i in range(max_iter):
    # for time step 0
    q_vals = actor.decision(state, cursor, env.end, [local_reward])
    if egreedy:
      action = np.random.choice(
          [np.argmax(q_vals.data.numpy()),
          np.random.randint(0,4)
          ], 
          p=eperc
        )
    else:
      action = np.argmax(q_vals.data.numpy())

    # for time step 1
    state1, cursor1 = env.get_state(state.copy(), cursor.copy(), optionswords[action])
    #local_reward1 = rwrd.distance(cursor1, env.end)
    local_reward1 = rwrd.dontstandstill(cursor1, env.end)
    #local_reward1 = rwrd.endonly(cursor1, env.end)
    #local_reward1 = rwrd.positivemoves(cursor1, env.end)

    q_vals1 = actor.decision(state1, cursor1, env.end, [local_reward1])
    if egreedy:
      action1 = np.random.choice(
          [np.argmax(q_vals1.data.numpy()),
          np.random.randint(0,4)
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
    print(cursor, local_reward, local_reward1)
    print(q_vals)
    print(q_vals1)
    loss = criterion(q_vals, target)
    loss_history = np.append(loss_history, loss.data.numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # step the game
    state = state1
    cursor = cursor1
    local_reward = local_reward1

    #if np.amin(cursor == env.end) == True:
    #  print('hit finishing pad')

    # TODO: do i need a diff target calc for the terminal state?

    # TODO: ha! I haven't specified an end of game. Potensch not a prob, since the reward will keep increasing as the agent steps off the end pad, and back on.

  if game % 1000 == 0:
    torch.save(actor.model.state_dict(), model_checkpoint)
    np.save('dat/loss_hist.npy', loss_history)

torch.save(actor.model.state_dict(), model_checkpoint)

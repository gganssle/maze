import numpy as np 
from env import blank 
from env import reward
from env import discount
from agnt import random, min
from plt import plotting_fools

plot = True
num_games = 3
max_iter = 20

actor = random.agent()
#actor = min.agent()
rwrd = reward.score()
dcount = discount.disc()

for game in range(num_games):
        env = blank.world(gridsize=10)        

        state = env.initial_state
        cursor = env.cursor
        local_reward = rwrd.initial_reward

        running_reward = []
        print('game number:', game)

        for i in range(max_iter):
                action = actor.decision(state, cursor, env.end, local_reward)

                state, cursor = env.get_state(state, cursor, action)

                local_reward = rwrd.endonly(cursor, env.end)
                running_reward.append(local_reward)

                if plot:
                        plotting_fools.plot1(state, running_reward, i)
                
                if np.amin(cursor == env.end) == True:
                        print(running_reward)
                        print(dcount.linear(running_reward))
                        break

                if i == max_iter - 1:
                        print(running_reward)
                        print(dcount.linear(running_reward))
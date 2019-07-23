import numpy as np 
from env import lumpy 
from env import reward
from env import discount
from agnt import rand, min, ffnn
from plt import plotting_fools

plot = True
num_games = 10
max_iter = 40

#actor = rand.agent()
#actor = min.agent()
actor = ffnn.agent(train=True)
rwrd = reward.score()
dcount = discount.disc()

for game in range(num_games):
        env = lumpy.world(gridsize=20)        

        state = env.initial_state
        cursor = env.cursor
        local_reward = rwrd.initial_reward

        running_reward = [0]
        print('game number:', game)


        for i in range(max_iter):
                action = actor.decision(state, cursor, env.end, running_reward)

                state, cursor = env.get_state(state, cursor, action)

                local_reward = rwrd.endonly(cursor, env.end)
                running_reward.append(local_reward)

                if plot:
                        plotting_fools.plot1(state, running_reward, i, cursor)
                
                if np.amin(cursor == env.end) == True:
                        print(running_reward)
                        print(dcount.expcoef(running_reward))
                        break

                if i == max_iter - 1:
                        print(running_reward)
                        print(dcount.expcoef(running_reward))

                
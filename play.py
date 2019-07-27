import numpy as np 
from env import lumpy, reward, discount
from agnt import rand, min, ffnn
from plt import plotting_fools

plot = True
num_games = 10
max_iter = 40

#actor1 = rand.agent()
#actor2 = min.agent()
actor = ffnn.agent()
rwrd = reward.score()
dcount = discount.disc()

for game in range(num_games):
        env = lumpy.world(gridsize=20)        

        state = env.initial_state
        cursor = env.cursor
        local_reward = rwrd.initial_reward

        running_reward = [0]
        discounted_reward = dcount.expcoef(running_reward)

        print('game number:', game)

        for i in range(max_iter):
                #actor = np.random.choice([actor1, actor2])
                actor = actor
                action = actor.decision(state, cursor, env.end, [local_reward])

                state, cursor = env.get_state(state, cursor, action)

                #local_reward = rwrd.dontstandstill(cursor, env.end)
                local_reward = rwrd.positivemoves(cursor, env.end)
                running_reward.append(local_reward)

                if plot:
                        plotting_fools.plot1(state, running_reward, i, cursor, action)
                
                if np.amin(cursor == env.end) == True:
                        print(running_reward)
                        break

                if i == max_iter - 1:
                        print(running_reward)
                
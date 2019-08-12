import numpy as np 
#from env import lumpy, reward, discount
from env import linear, reward, discount
from agnt import rand, min, ffnn, linear_ffnn
from plt import plotting_fools, linear_plotting_tools

plot = False
num_games = 1000
max_iter = 5
win_hist = []

#actor1 = rand.agent()
#actor2 = min.agent()
#actor = ffnn.agent()
actor = linear_ffnn.agent()
rwrd = reward.score()
dcount = discount.disc()

for game in range(num_games):
        #env = lumpy.world(gridsize=10)        
        env = linear.world()

        state = env.initial_state
        cursor = env.cursor
        local_reward = rwrd.initial_reward

        running_reward = [0]
        discounted_reward = dcount.expcoef(running_reward)

        for i in range(max_iter):
                #actor = np.random.choice([actor1, actor2])
                actor = actor
                action = actor.decision(state, cursor, env.end, [local_reward])

                state, cursor = env.get_state(state, cursor, action)

                local_reward = rwrd.dontstandstill(cursor, env.end)
                #local_reward = rwrd.positivemoves(cursor, env.end)
                running_reward.append(local_reward)

                if plot:
                        #plotting_fools.plot1(state, running_reward, i, cursor, action)
                        linear_plotting_tools.plot1(state, running_reward, i, cursor, action)
                        
                if np.amin(cursor == env.end) == True:
                        win_hist.append(1)
                        break

                if i == max_iter - 1:
                        win_hist.append(0)
           
win_hist = np.array(win_hist)
win_perc = 100 * np.nonzero(win_hist)[0].shape[0] / num_games

print(f'agent won {win_perc}% of the games')
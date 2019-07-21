import numpy as np 
from env import blank 
from env import reward
from agnt import random 
from plt import plotting_fools

plot = True

env = blank.world()
actor = random.agent()
rwrd = reward.score()

state = env.initial_state
cursor = env.cursor
local_reward = rwrd.initial_reward

running_reward = []

for i in range(10000):
    action = actor.decision(state, local_reward)

    state, cursor = env.get_state(state, cursor, action)

    local_reward = rwrd.endonly(cursor, env.end)
    running_reward.append(local_reward)

    if plot:
        plotting_fools.plot1(state, running_reward, i)
        
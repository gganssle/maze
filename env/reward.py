import numpy as np 

class score(object):
    def __init__(self):
        self.initial_reward = 0 

    def endonly(self, cursor, end):
        if np.amin(cursor == end) == True:
            reward = 1
        else:
            reward = 0

        return reward

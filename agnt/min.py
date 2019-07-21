import numpy as np 

class agent(object):
    def __init__(self):
        pass 

    def decision(self, state, cursor, end, reward):
        current_distance = np.linalg.norm(end - cursor)

        options = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        optionswords = ['left', 'right', 'up', 'down']

        action = np.argmin(np.linalg.norm(end - (cursor+options), axis=1))

        return optionswords[action]
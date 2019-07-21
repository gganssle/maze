import numpy as np 

class agent(object):
    def __init__(self):
        pass

    def decision(self, state, reward):
        return np.random.choice(['left', 'right', 'up', 'down'])
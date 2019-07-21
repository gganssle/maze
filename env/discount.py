import numpy as np 

class disc(object):
    def __init__(self):
        pass 

    def linear(self, reward_history):
        reward_history = np.array(reward_history)

        discounted = np.linspace(0, reward_history[-1], reward_history.shape[0])

        return discounted
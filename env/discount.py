import numpy as np 

class disc(object):
    def __init__(self):
        pass 

    def linear(self, reward_history):
        reward_history = np.array(reward_history)

        discounted = np.linspace(0, reward_history[-1], reward_history.shape[0])

        return discounted

    def expcoef(self, reward_history, discount_factor=0.9):
        reward_history = np.array(reward_history)

        discounted = [discount_factor**i * reward_history[i] for i in range(reward_history.shape[0])]
        discounted = np.array(discounted)

        return discounted
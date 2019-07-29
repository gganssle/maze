import numpy as np 

class env(object):
  def __init__(self):
    pass 
  
  def get_state(self):
    idx = np.random.choice([0,1])

    state = np.zeros(2)
    state[idx] = 1

    return state 
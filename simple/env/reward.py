class score(object):
  def __init__(self):
    pass 

  def vanilla(self, choice, state):
    if state[choice] == 1:
      return 1
    if state[choice] == 0:
      return 0
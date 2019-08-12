import numpy as np 

class world(object):
    def __init__(self):
        self.end = [np.random.choice([0,1,3,4])]
        self.start =  [2]
        self.cursor = self.start

        self.initial_state = np.zeros(5)

        self.initial_state[self.start] = 1
        self.initial_state[self.end] = 2

        
    def get_state(self, state, cursor, action):
        tmpcursor = cursor.copy()

        # take action
        if action == 'sit':
            tmpcursor = cursor
        elif action == 'right':
            tmpcursor[0] += 1
        elif action == 'left':
            tmpcursor[0] -= 1
        else:
            raise ValueError

        # ensure cursor is on the playing field
        if (tmpcursor[0] < 4) & (tmpcursor[0] >= 0):
            cursor = tmpcursor

        # write an action trail
        #state[cursor[0], cursor[1]] += 1

        return state, cursor
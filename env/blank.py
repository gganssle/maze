import numpy as np 

class world(object):
    def __init__(self, gridsize=10):
        self.gridsize = gridsize

        self.end = np.random.randint(0, self.gridsize, (2))
        self.start =  np.random.randint(0, self.gridsize, (2))
        self.cursor = self.start

        self.initial_state = np.zeros((self.gridsize, self.gridsize))

        self.initial_state[self.start[0], self.start[1]] = 1
        self.initial_state[self.end[0], self.end[1]] = 2

    def get_state(self, state, cursor, action):
        tmpcursor = cursor.copy()

        if action == 'left':
            tmpcursor[0] += 1
        elif action == 'right':
            tmpcursor[0] -= 1
        elif action == 'up':
            tmpcursor[1] += 1
        elif action == 'down':
            tmpcursor[1] -= 1
        else:
            raise ValueError

        if (tmpcursor[0] < self.gridsize) & (tmpcursor[0] >= 0):
            cursor[0] = tmpcursor[0]
        if (tmpcursor[1] < self.gridsize) & (tmpcursor[1] >= 0):
            cursor[1] = tmpcursor[1]

        state[cursor[0], cursor[1]] += .2

        return state, cursor
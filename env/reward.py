import numpy as np 

class score(object):
    def __init__(self):
        self.initial_reward = 0
        self.oldcursor = [0,0]
        self.oldreward = 0

    def endonly(self, cursor, end):
        if np.amin(cursor == end) == True:
            reward = 1
        else:
            reward = 0

        return reward

    def dontstandstill(self, cursor, end):
        if np.amin(cursor == end) == True:
        #if (np.amin(cursor == end) == True) | (np.amin(cursor == [1,1]) == True) | (np.amin(cursor == [2,2]) == True) | (np.amin(cursor == [3,3]) == True) | (np.amin(cursor == [4,4]) == True) | (np.amin(cursor == [4,4]) == True) | (np.amin(cursor == [5,5]) == True) | (np.amin(cursor == [6,6]) == True) | (np.amin(cursor == [7,7]) == True) | (np.amin(cursor == [8,8]) == True) | (np.amin(cursor == [9,9]) == True) | (np.amin(cursor == [10,10]) == True) | (np.amin(cursor == [11,11]) == True) | (np.amin(cursor == [12,12]) == True) | (np.amin(cursor == [13,13]) == True) | (np.amin(cursor == [14,14]) == True) | (np.amin(cursor == [15,15]) == True) | (np.amin(cursor == [16,16]) == True) | (np.amin(cursor == [17,17]) == True):
            reward = 1
        elif np.amin(cursor == self.oldcursor) == True:
            #reward = self.oldreward + -0.05 #increased historical feedback
            reward = -0.01
            self.oldreward = reward
        else:
            reward = 0
            self.oldreward = 0
        
        self.oldcursor = cursor.copy()

        return reward

    def positivemoves(self, cursor, end):
        if np.amin(cursor == end) == True:
            reward = 1
        elif np.amin(cursor == self.oldcursor) == False:
            reward = 0.01
        else:
            reward = 0

        self.oldcursor = cursor.copy()

        return reward

    def distance(self, cursor, end):
        reward = np.linalg.norm(cursor - end)
        reward = 1 / reward

        return reward
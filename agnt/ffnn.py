import numpy as np 
import torch 
import torch.nn as nn


class net(nn.Module):
            def __init__(self):
                super(net, self).__init__()
                self.dense1 = nn.Linear(400+2+2+1, 200)
                self.relu1 = nn.LeakyReLU(True)
                self.dense2 = nn.Linear(200, 50)
                self.relu2 = nn.LeakyReLU(True)
                self.dense3 = nn.Linear(50, 4)

            def forward(self, x):
                x = self.relu1(self.dense1(x))
                x = self.relu2(self.dense2(x))
                x = self.dense3(x)

                return x


class agent(object):
    def __init__(self, train=False):
        model_checkpoint = '../dat/model'
        self.train = train
        self.discount_factor = 0.9

        self.model = net()

        if self.train == False:
            self.model.load_state_dict(torch.load(model_checkpoint))
            self.model.eval()


    def decision(self, state, cursor, end, reward):
        optionswords = ['left', 'right', 'up', 'down']

        if self.train == False:
            total_state = torch.cat((
                torch.Tensor(state.flatten()),
                torch.Tensor(cursor),
                torch.Tensor(end),
                torch.Tensor(reward)  
            ))

            q_vals = self.model(total_state)

        else:
            

        action = np.argmax(q_vals)

        return optionswords[action]


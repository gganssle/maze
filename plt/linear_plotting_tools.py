import numpy as np 
import matplotlib.pyplot as plt 

def plot1(state, running_reward, step, cursor, action):
    fig, ax = plt.subplots(2, 1)

    ax[0].set_title(f'step {step}, action {action}')
    im = ax[0].imshow(np.transpose(np.expand_dims(state, axis=1)))
    ax[0].grid()
    plt.colorbar(im, ax=ax[0])
    ax[0].scatter(cursor[0], 0, color='red')

    cumreward = np.cumsum(np.array(running_reward))
    #ax[1].plot(running_reward)
    ax[1].plot(cumreward)

    plt.show(block=False)
    plt.pause(.05)
    plt.close()
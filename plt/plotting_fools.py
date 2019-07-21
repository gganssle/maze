import numpy as np 
import matplotlib.pyplot as plt 

def plot1(state, running_reward, step):
    fig, ax = plt.subplots(2, 1)

    ax[0].set_title(f'step {step}')
    im = ax[0].imshow(state)
    ax[0].grid()
    plt.colorbar(im, ax=ax[0])

    ax[1].plot(running_reward)

    plt.show(block=False)
    plt.pause(.1)
    plt.close()
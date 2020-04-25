import numpy as np
import matplotlib.pyplot as plt

def find_optimum_price(prices, p, adv_rew, graphics=False):
    tot_rew = []
    '''
    Chissà perché capiti di non prendere una bid
    '''
    # In case no bid has been chosen
    for a in adv_rew:
        if a == []:
            a = 0

    for price in prices:
        tot_rew.append(price*(sum([p[i+1](price)*adv_rew[i] for i in range(3)])))

    if graphics:
        plt.figure(1)
        plt.plot(tot_rew)
        plt.show()

    opt = prices[np.argmax(tot_rew)]

    return [opt, max(tot_rew)]
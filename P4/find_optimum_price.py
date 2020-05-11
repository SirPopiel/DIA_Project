import numpy as np
import matplotlib.pyplot as plt


def find_optimum_price(prices, p, adv_rew, graphics=False):
    tot_rew = []

    for price in prices:
        tot_rew.append(price*(sum([p[i+1](price)*adv_rew[i] for i in range(3)])))

    if graphics:
        plt.figure(1)
        plt.plot(prices, tot_rew)
        plt.xlabel("Price")
        plt.ylabel("Expected reward")
        plt.title("Expected reward given a price")

    opt = prices[np.argmax(tot_rew)]
    opt_rew = [opt * p[i+1](opt) for i in range(3)]

    return [opt, opt_rew]

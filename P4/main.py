import numpy as np
import matplotlib.pyplot as plt
from bidding_environment import *
from MovingBiddingEnvironment import *
from pricing_environment import *
from ts_learner import *
from gpts_learner import *
from SlidingWindowsGPTS_Learner import *
from bid_optimizer import *
from ts_learning_subcampaign import *
from find_optimum_price import *

verbose = True
graphics = True
debug = True  # if True it shows useless plots
n_arms_adv = 50
time_horizon = 30  # time used for optimizing the bids
window_size = 30

min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms_adv)
sigma = 10

np.random.seed(0)
# bid_optimizer returns the number of clicks obtained solving the optimization problem
adv_rew = bid_optimizer(bids, n_arms_adv, sigma, time_horizon,
                        sliding_window = False, window_size = window_size,
                        verbose=verbose, graphics=graphics)
# adv_rew = [75.88295209998937, 57.01093426334284, 76.22638762838052]           #solo comodo per non runnare bid_optimizer quando si provano le altre parti
# todo: in fase finale rimuovere questo

n_arms_pricing = 20
price_min = 50
price_max = 70
prices = np.linspace(price_min, price_max, n_arms_pricing)

# Probabilities of conversion given a price
p = {
    1: (lambda x: (0.7 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    2: (lambda x: (0.9 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    3: (lambda x: (0.5 * np.exp(-(x - 50) ** (1 / 2) / 20)))
}

collected_rewards_subcampaign = [[] for i in range(3)]
for i in range(3):
    collected_rewards_subcampaign[i] = \
        ts_learning_subcampaign(n_arms=n_arms_pricing, prices=prices,
                                subcampaign=i + 1, time_horizon=time_horizon, p=p)

if debug:
    plt.figure(0)
    plt.ylabel("Collected rewards during ts_learning pricing")
    plt.xlabel("t")
    plt.plot(collected_rewards_subcampaign[0], 'r.')
    plt.plot(collected_rewards_subcampaign[1], 'b.')
    plt.plot(collected_rewards_subcampaign[2], 'k.')
    plt.legend(["Sub1", "Sub2", "Sub3"])

[best_price, opt_rew] = find_optimum_price(prices=prices, p=p, adv_rew=adv_rew, graphics=debug)

if verbose:
    print("Best price is: ", best_price)

if graphics:
    plt.figure(2)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(opt_rew[0] - collected_rewards_subcampaign[0]), 'r', label="Sub1")
    plt.plot(np.cumsum(opt_rew[1] - collected_rewards_subcampaign[1]), 'b', label="Sub2")
    plt.plot(np.cumsum(opt_rew[2] - collected_rewards_subcampaign[2]), 'k', label="Sub3")
    plt.legend()
    plt.title("Cumulative regret of each subcampaign learning the best price")

regret_allsubcampaigns = []
for i in range(3):
    regret_allsubcampaigns.append(sum(regret_allsubcampaigns) * np.ones(time_horizon)
                                  + opt_rew[i] - collected_rewards_subcampaign[i])
if graphics:
    plt.figure()
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.plot(np.cumsum(regret_allsubcampaigns), label='Cumulative regret learning the best price')
    plt.legend()
    plt.show()

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
n_arms_adv = 25
time_horizon = 150
window_size = 30

min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms_adv)
sigma = 10

np.random.seed(0)
# bid_optimizer returns the number of clicks obtained solving the optimization problem
adv_rew = bid_optimizer(bids, n_arms_adv, sigma, time_horizon,
                        sliding_window=True, window_size=window_size,
                        verbose=verbose, graphics=graphics)
# adv_rew = [28.31, 67.86, 56.53]           #solo comodo per non runnare bid_optimizer quando si provano le altre parti

n_arms_pricing = 20
price_min = 50
price_max = 70
prices = np.linspace(price_min, price_max, n_arms_pricing)

collected_rewards_subcampaign = [[] for i in range(3)]
for i in range(3):
    collected_rewards_subcampaign[i] = \
        ts_learning_subcampaign(n_arms=n_arms_pricing, prices=prices,
                                subcampaign=i+1, time_horizon=time_horizon)

if graphics:
    plt.figure(0)
    plt.ylabel("Rewards")
    plt.xlabel("t")
    plt.plot(collected_rewards_subcampaign[0], 'r.')
    plt.plot(collected_rewards_subcampaign[1], 'b.')
    plt.plot(collected_rewards_subcampaign[2], 'k.')
    plt.legend(["Sub1", "Sub2", "Sub3"])
    plt.show()

[best_price, opt_rew] = find_optimum_price(prices=prices, p=p, adv_rew=adv_rew, graphics=graphics)

if verbose:
    print("Best price is: ", best_price)

if graphics:
    plt.figure(2)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(opt_rew - collected_rewards_subcampaign[0]), 'r')
    plt.plot(np.cumsum(opt_rew - collected_rewards_subcampaign[1]), 'b')
    plt.plot(np.cumsum(opt_rew - collected_rewards_subcampaign[2]), 'k')
    plt.legend(["Sub1", "Sub2", "Sub3"])
    plt.show()
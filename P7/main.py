import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/federico/Documents/GATTI/DIA_Project/Restart/')

import numpy as np
import matplotlib.pyplot as plt
from bidding_environment import *
from pricing_environment import *
from ts_learner import *
from gpts_learner import *
from data import p
from good_knapsack import *
import math
import pulp


# Sets initial data
n_arms_ads = 25 # number of arms for advertising
n_arms_pricing = 10 # number of arms for pricing

T = 100 # T for Times
min_bid = 0.0
max_bid = 1.0
scale_factor_for_bid = 10
bids = np.linspace(min_bid, max_bid, n_arms_ads) # bids are a linspace
sigma = 10

# Presets regrets and rewards that are going to be computed
regrets_per_subcampaign = []
rewards_per_subcampaign = []

price_min = 50.0
price_max = 70.0
prices = np.linspace(price_min, price_max, n_arms_pricing)


ad_envs = [BiddingEnvironment(bids, sigma, subcampaign=subcampaign) for subcampaign in [1,2,3]]
meta_gpts_learners = [GPTS_Learner(n_arms=n_arms_ads, arms=bids) for subcampaign in [1,2,3]]
allocations = [
    {1: 10, 2:20, 3:10}
]

pricing_envs = [PricingEnvironment(n_arms=n_arms_pricing, prices=prices, p=p, subcampaign=subcampaign) for subcampaign in [1,2,3]]
ts_learner = TS_Learner(n_arms=n_arms_pricing)
for t in range(T):
    # 3 subcampaigns:
    rewards_per_subcampaign = []
    print(allocations)
    for subcampaign in [1, 2, 3]:
        ad_bid_to_try = allocations[-1][subcampaign] # pull the allocated arm
        n_clicks = ad_envs[subcampaign-1].round(ad_bid_to_try) # gets another random value from it
        rewards_from_ad = 0
        # TODO: Change order of users' trials
        for t in range(int(n_clicks)):
            price_to_try = ts_learner.pull_arm()
            reward = pricing_envs[subcampaign-1].round(price_to_try)
            rewards_from_ad += reward#*price_to_try
            ts_learner.update(price_to_try, reward, price_to_try)
        rewards_from_ad -= 0#bids[ad_bid_to_try]*scale_factor_for_bid
        meta_gpts_learners[subcampaign-1].update(ad_bid_to_try, rewards_from_ad) # updates the learner
        # Appends to the rewards the values at lower CI
        rewards_per_subcampaign.append(np.random.normal(meta_gpts_learners[subcampaign-1].means, meta_gpts_learners[subcampaign-1].sigmas))
    allocations.append(good_knapsack(bids, rewards_per_subcampaign, 1.0))

print(allocations)
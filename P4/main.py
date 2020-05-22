import numpy as np
from pricing_single_learner import *
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
n_arms_adv = 10
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

#################################
# Loads the Pricing environment

n_clicks = [120, 100, 40] # clicks for optimal budget allocation
time_horizon = 150
n_arms = 20
price_min = 50
price_max = 70
list_prices = np.linspace(price_min, price_max, n_arms)
conversion_rate_curve = [ # Probabilities of conversion given a price
    (lambda x: (0.7 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    (lambda x: (0.9 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    (lambda x: (0.5 * np.exp(-(x - 50) ** (1 / 2) / 20)))
]

# Defines the pricing env
env = PricingEnvironment(n_clicks, n_arms, list_prices, time_horizon, conversion_rate_curve)

# Gets the learner
learnerP4 = pricing_single_learner(env)

# Lets it work
[opt_price, opt_reward] = learnerP4.find_optimal_price(graphics = True)

print("Best price is: ", opt_price)

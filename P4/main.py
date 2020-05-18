import numpy as np
from pricing_single_learner import *
from pricing_environment import *

np.random.seed(0)

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

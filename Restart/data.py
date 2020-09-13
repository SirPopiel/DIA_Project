import numpy as np
import time



verbose = True
graphics = True
debug = True  # if True it shows useless plots
sliding_window = False # if True the experiment is conducted with time-varying demand curves
n_experiments = 5
adv_budget = 1.0
n_arms_adv = 25
#n_arms_adv = [9,17,25,33]
# in order to compare graphically multiple instances, multiple arms can be tested

time_horizon = 200  # time used for optimizing the budget allocation
window_size = int(time_horizon/10)

min_budget = 0.0
max_budget = 1.0
budgets =[]

if sliding_window:
    graphics = False

if isinstance(n_arms_adv, list):
    for i in range(len(n_arms_adv)):
        budgets.append(np.linspace(min_budget, max_budget, n_arms_adv[i]))
else:
    budgets = [np.linspace(min_budget, max_budget, n_arms_adv)]


sigma = 10
n_for_b = None
n_proportion_phases = None

ad_pricing_range_max = {
    1: 3000,
    2: 1000,
    3: 100
} # from 0 -> 1 to 0 -> range_max

if sliding_window:
    # Functions that assign the number of clicks to a given budget
    # They are monotone increasing in [0,1]
    n_for_b = {
        1: [(lambda x: 100 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3))),
            (lambda x: 20 * (1.0 - np.exp(-4 * x + 1 * x ** 3)))
            ],
        2: [(lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3))),
            (lambda x: 20 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-4 * x + 1 * x ** 3)))
            ],
        3: [(lambda x: 20 * (1.0 - np.exp(-4 * x + 1 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3)))
            ]
    }

    # Proportion of the time horizon in which each phase takes place
    n_proportion_phases = {
        #1: [0, 0.3, 0.6],
        #2: [0, 0.3, 0.5],
        #3: [0, 0.2, 0.6]
        1: [0, 0.2, 0.6],
        2: [0, 0.2, 0.6],
        3: [0, 0.2, 0.6]
    }
    # Proportion of time horizon where you have an actually different environment
    #abrupt_phases = [0, 0.2, 0.3, 0.5, 0.6]
    abrupt_phases = [0, 0.2, 0.6]
else:
    # Functions that assign the number of clicks to a given budget
    # They are monotone increasing in [0,1]
    n_for_b = {
        1: (lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3))),
        2: (lambda x: 75 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
        3: (lambda x: 100 * (1.0 - np.exp(-4 * x + 1 * x ** 3)))
    }

    n_proportion_phases = None
    abrupt_phases = None

'''
n_arms_pricing = 20
price_min = 50
price_max = 70
prices = np.linspace(price_min, price_max, n_arms_pricing)

# Probabilities of conversion given a price
p = {
    1: (lambda x: (0.7 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    2: (lambda x: (0.9 * np.exp(-(x - 50) ** (1 / 2) / 20))),
    3: (lambda x: (0.5 * np.exp(-(x*x*x - 50) ** (1 / 2) / 50)))
}

# Expected price of selling given price
p_star = {
    1: (lambda x: (x-price_min)/(price_max-price_min) * p[1](x)),
    2: (lambda x: (x-price_min)/(price_max-price_min) * p[2](x)),
    3: (lambda x: (x-price_min)/(price_max-price_min) * p[3](x)),
}
'''

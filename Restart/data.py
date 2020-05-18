import numpy as np


verbose = True
graphics = True
debug = True  # if True it shows useless plots
n_arms_adv = 25
time_horizon = 150  # time used for optimizing the bids
window_size = 30

min_budget = 0.0
max_budget = 1.0
budgets = np.linspace(min_budget, max_budget, n_arms_adv)
sigma = 10

n_to_f = {
    1: (lambda x: 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3))),
    2: (lambda x: 75 * (1.0 - np.exp(-5*x + 2*x**3))),
    3: (lambda x: 100 * (1.0 - np.exp(-4*x + 1*x**3)))
}

# Functions that assign the number of clicks to a given bid
# They are monotone increasing in [0,1]
n_t_to_f = {
    1: [(lambda x: 100 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x: 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3))),
        (lambda x: 75 * (1.0 - np.exp(-4*x + 1*x**3)))
    ],
    2: [(lambda x: 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3))),
        (lambda x: 75 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x: 100 * (1.0 - np.exp(-4*x + 1*x**3)))
    ],
    3: [(lambda x: 75 * (1.0 - np.exp(-4*x + 1*x**3))),
        (lambda x: 100 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x: 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3)))
    ]
}

# Proportion of the time horizon in which each phase takes place
n_proportion_phases = {
    1: [0, 0.3, 0.6],
    2: [0, 0.3, 0.5],
    3: [0, 0.2, 0.6]
}



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

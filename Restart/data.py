import numpy as np
from budget_optimizer import *
import time

verbose = True
graphics = True
debug = True  # if True it shows useless plots
sliding_window = False
n_experiments = 10
adv_budget = 1.0
n_arms_adv = 20
time_horizon = 150  # time used for optimizing the budget allocation
window_size = time_horizon/6

min_budget = 0.0
max_budget = 1.0
budgets = np.linspace(min_budget, max_budget, n_arms_adv)
sigma = 10

if sliding_window:
    # Functions that assign the number of clicks to a given budget
    # They are monotone increasing in [0,1]
    n_for_b = {
        1: [(lambda x: 100 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3))),
            (lambda x: 75 * (1.0 - np.exp(-4 * x + 1 * x ** 3)))
            ],
        2: [(lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3))),
            (lambda x: 75 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-4 * x + 1 * x ** 3)))
            ],
        3: [(lambda x: 75 * (1.0 - np.exp(-4 * x + 1 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + 2 * x ** 3))),
            (lambda x: 100 * (1.0 - np.exp(-5 * x + x ** 2 + 1 * x ** 3)))
            ]
    }

    # Proportion of the time horizon in which each phase takes place
    n_proportion_phases = {
        1: [0, 0.3, 0.6],
        2: [0, 0.3, 0.5],
        3: [0, 0.2, 0.6]
    }
    # Proportion of time horizon where you have an actually different environment
    abrupt_phases = [0, 0.2, 0.3, 0.5, 0.6]

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


def main():
    start_time = time.time()
    budget_optimizer(budget=adv_budget, list_budgets=budgets, sigma=sigma, time_horizon=time_horizon,
                     sliding_window=sliding_window, abrupt_phases=abrupt_phases, n_experiments=n_experiments,
                     graphics=graphics, verbose=verbose)
    print("\nTotal completion time: \n" + "--- %.2f seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()

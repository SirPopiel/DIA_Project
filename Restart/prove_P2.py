import matplotlib.pyplot as plt
from data import *
from budget_optimizer import budget_optimizer

budget_optimizer(budget = 40, list_budgets = budgets, sigma = sigma, time_horizon = 20, sliding_window = sliding_window,
                 n_experiments = 10 , graphics = graphics, verbose = verbose)
plt.show()

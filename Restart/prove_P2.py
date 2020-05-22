import matplotlib.pyplot as plt
from data import *
from budget_optimizer import budget_optimizer

budget_optimizer(budget = 1, list_budgets=budgets, sigma=sigma, time_horizon = 50, n_experiments = 3, graphics=graphics)
plt.show()

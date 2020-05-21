import matplotlib.pyplot as plt
from data import *
from budget_optimizer import budget_optimizer

budget_optimizer(budget=1.2, list_budgets=budgets, sigma=sigma, time_horizon=60, n_experiments=10, graphics=graphics)
plt.show()
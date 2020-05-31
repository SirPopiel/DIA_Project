import matplotlib.pyplot as plt
from data import *
from budget_optimizer import budget_optimizer

<<<<<<< HEAD
budget_optimizer(budget = 40, list_budgets = budgets, sigma = sigma, time_horizon = 20, sliding_window = sliding_window,
                 n_experiments = 10 , graphics = graphics, verbose = verbose)
=======
budget_optimizer(budget=1, list_budgets=budgets, sigma=sigma, time_horizon=100, sliding_window=sliding_window,
                 abrupt_phases=abrupt_phases, n_experiments=10, graphics=graphics, verbose=verbose)
>>>>>>> 7e076a19216e621a5ff21e1f5db8ce1c9e2b264d
plt.show()

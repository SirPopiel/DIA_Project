import matplotlib.pyplot as plt
from data import *
from budget_optimizer import budget_optimizer


ad, regrets = budget_optimizer(budget = 1, budgets = budgets, sigma = sigma, time_horizon= time_horizon, sliding_window = sliding_window, window_size = window_size,
                abrupt_phases = abrupt_phases, n_experiments = n_experiments, graphics = graphics, verbose = verbose, tune = False)


# per plottare la differenza tra sliding window e non sliding window gpts
if sliding_window:
    graphics = False
    ad2, regrets2 = budget_optimizer(budget = 1, budgets = budgets, sigma = sigma, time_horizon= time_horizon, sliding_window = sliding_window, window_size = time_horizon,
                    abrupt_phases = abrupt_phases, n_experiments = n_experiments, graphics = graphics, verbose = verbose, tune = False)


    for iteration in range(len(budgets)):
        plt.plot(np.cumsum(np.mean(regrets[iteration], axis=0)), 'b', label='%s window GPTS' % window_size)
        plt.plot(np.cumsum(np.mean(regrets2[iteration], axis=0)),'k', label='Standard GPTS')
    if sliding_window:
        plt.vlines([i * time_horizon for i in abrupt_phases], 0,8000, colors='r', linestyles='solid', label = 'Phase change')
    plt.legend()


plt.show()

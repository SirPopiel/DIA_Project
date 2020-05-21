import numpy as np
import matplotlib.pyplot as plt
from bidding_environment import *
from moving_bidding_environment import *
from gpts_learner import *
from SlidingWindowsGPTS_Learner import *
from good_knapsack import *
from optimization import dynamic_opt

def budget_optimizer(budget, list_budgets, sigma, time_horizon, n_tuning=20, n_experiments=1,
                     sliding_window=False, window_size=0, graphics=False, verbose=False):
    n_arms = len(list_budgets)
    rewards_per_subcampaign_per_experiment = [[] for _ in range(3)]
    budget_index = np.max(np.argwhere(list_budgets <= budget))

    for e in range(n_experiments):
        envs = []
        regrets_per_subcampaign = []
        rewards_per_subcampaign = [[] for _ in range(3)]
        gpts_learner = []

        for subcampaign in [1, 2, 3]:
            if sliding_window:
                envs.append(MovingBiddingEnvironment(budgets=list_budgets, sigma=sigma, time_horizon=time_horizon,
                                                     subcampaign=subcampaign))
                gpts_learner.append(SlidingWindowsGPTS_Learner(n_arms=n_arms, arms=list_budgets,
                                                               window_size=window_size))
            else:
                envs.append(BiddingEnvironment(budgets=list_budgets, sigma=sigma, subcampaign=subcampaign))
                gpts_learner.append(GPTS_Learner(n_arms=n_arms, arms=list_budgets))

            # Tuning hyperparameters of the gps
            # In order to do this we assume at this point we know the curves which characterize our environments
            x_real = np.linspace(min(list_budgets), max(list_budgets), n_tuning)
            if not sliding_window:
                y_real = n_for_b[subcampaign](x_real)
            else:
                y_real = n_for_b[subcampaign][1](x_real)
            x_real = np.atleast_2d(x_real).T
            gpts_learner[subcampaign - 1].gp.fit(x_real, y_real)

        # Initializing the budget allocation evenly
        budget_allocation = np.zeros(3)
        starting_allocation = np.max(list_budgets[np.argwhere(list_budgets <= budget / 3)]) * np.ones(3)

        for subcampaign in [1, 2, 3]:
            budget_allocation[subcampaign - 1] = \
                np.max(list_budgets[np.argwhere(list_budgets <= starting_allocation[subcampaign - 1])])

        # Starting each experiment
        for t in range(time_horizon):
            for subcampaign in [1, 2, 3]:
                pulled_arm = gpts_learner[subcampaign - 1].pull_arm(budget_allocation[subcampaign - 1])
                reward = envs[subcampaign - 1].round(pulled_arm)
                gpts_learner[subcampaign - 1].update(pulled_arm, reward)
                rewards_per_subcampaign[subcampaign - 1] = gpts_learner[subcampaign - 1].means

            # new_allocation = good_knapsack(list_budgets, rewards_per_subcampaign, budget)
            budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                            rewards_per_subcampaign=rewards_per_subcampaign)
            # for subcampaign in [1, 2, 3]:
            #     budget_allocation[subcampaign - 1] = \
            #         np.max(list_budgets[np.argwhere(list_budgets <= new_allocation[subcampaign - 1])])

        for subcampaign in [1, 2, 3]:
            rewards_per_subcampaign_per_experiment[subcampaign - 1].append(gpts_learner[subcampaign - 1].means)

        # if graphics:
        #     x_pred = np.atleast_2d(list_budgets).T
        #     for i in range(3):
        #         y_pred, sigma = gpts_learner[i].gp.predict(x_pred, return_std=True)
        #         plt.figure()
        #         if not sliding_window:
        #             x = np.linspace(np.min(list_budgets), np.max(list_budgets), 100)
        #             plt.plot(x, n_for_b[i + 1](x), 'r', label='Real function')
        #         plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
        #         plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
        #                  np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
        #                  alpha=.5, fc='b', ec='None', label='95% Confidence Interval')
        #         plt.xlabel('$Budget$')
        #         plt.ylabel('$Number of clicks$')
        #         plt.title("Check on the learned curves")
        #         plt.legend(loc='lower right')
    if graphics:
        for i in range(3):
            plt.figure()
            if not sliding_window:
                x = np.linspace(np.min(list_budgets), np.max(list_budgets), 100)
                plt.plot(x, n_for_b[i + 1](x), 'r', label='Real function')
            plt.plot(list_budgets, np.mean(rewards_per_subcampaign_per_experiment[subcampaign - 1], axis=0), 'b.')
            plt.xlabel('Budget')
            plt.ylabel('Number of clicks')

    adv_rew = np.zeros(3)

    return adv_rew
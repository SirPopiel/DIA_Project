import numpy as np
import matplotlib.pyplot as plt
import random
from bidding_environment import *
from moving_bidding_environment import *
from gpts_learner import *
from SlidingWindowsGPTS_Learner import *
from good_knapsack import *
from optimization import dynamic_opt


def t_to_phase(subcampaign, time_horizon, t_):
    for p in range(3):
        if n_proportion_phases[subcampaign][p] * time_horizon >= t_:
            return p
    return 0


def budget_optimizer(budget, list_budgets, sigma, time_horizon, n_tuning=25, n_experiments=1,
                     sliding_window=False, window_size=0, abrupt_phases=None, graphics=False, verbose=False):
    if abrupt_phases is None:
        abrupt_phases = []
    n_arms = len(list_budgets)
    rewards_per_subcampaign_per_experiment = [[[] for _ in range(n_experiments)] for _ in range(3)]
    regrets_per_experiment = [[] for _ in range(n_experiments)]
    budget_index = np.max(np.argwhere(list_budgets <= budget))

    # Computing the optimal budget allocation in the case of stationary environment
    if not sliding_window:
        real_rewards = [[] for _ in range(3)]
        for subcampaign in [1, 2, 3]:
            real_rewards[subcampaign-1] = n_for_b[subcampaign](list_budgets)
        optimal_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                                rewards_per_subcampaign=real_rewards)
        optimal_click = sum([n_for_b[i+1](optimal_budget_allocation[i]) for i in range(3)])
        if verbose:
            print("The best budget allocation would have been: ", optimal_budget_allocation)
            print("with corresponding number of clicks: ", [n_for_b[i+1](optimal_budget_allocation[i]) for i in range(3)])

    for e in range(n_experiments):
        envs = []
        collected_rewards_per_subcampaign = np.zeros(3)
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
                y_real = n_for_b[subcampaign][0](x_real)
            x_real = np.atleast_2d(x_real).T
            gpts_learner[subcampaign - 1].gp.fit(x_real, y_real)

        # Initializing the budget allocation evenly
        # budget_allocation = np.max(list_budgets[np.argwhere(list_budgets <= budget / 3)]) * np.ones(3)

        # Initializing the budget allocation randomly
        budget_sub1 = random.choice(list_budgets)
        budget_sub2 = random.choice(list_budgets[np.argwhere(list_budgets <= budget - budget_sub1)])
        budget_sub3 = random.choice(list_budgets[np.argwhere(list_budgets <= budget - budget_sub1 - budget_sub2)])
        budget_allocation = [budget_sub1, budget_sub2, budget_sub3]

        if sliding_window:
            last_phase = 0

        # Starting each experiment
        for t in range(time_horizon):
            # Computing the optimal budget allocation in the case of non-stationary environment
            if sliding_window and last_phase < len(abrupt_phases) and abrupt_phases[last_phase]*time_horizon >= t:
                real_rewards = [[] for _ in range(3)]
                for subcampaign in [1, 2, 3]:
                    phase_sub = t_to_phase(subcampaign, time_horizon, t)
                    real_rewards[subcampaign - 1] = n_for_b[subcampaign][phase_sub](list_budgets)
                optimal_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                                        rewards_per_subcampaign=real_rewards)
                optimal_click = sum([n_for_b[i + 1][phase_sub](optimal_budget_allocation[i]) for i in range(3)])
                # print("The best budget allocation would have been: ", optimal_budget_allocation)
                # print("with corresponding number of clicks: ",
                #       [n_for_b[i + 1][phase_sub](optimal_budget_allocation[i]) for i in range(3)])
                last_phase += 1

            # Starting exploration of our environments
            for subcampaign in [1, 2, 3]:
                pulled_arm = gpts_learner[subcampaign - 1].pull_arm(budget_allocation[subcampaign - 1])
                reward = envs[subcampaign - 1].round(pulled_arm)
                collected_rewards_per_subcampaign[subcampaign - 1] = reward
                gpts_learner[subcampaign - 1].update(pulled_arm, reward)
                gpts_learner[subcampaign - 1].means[0] = 0          # Already know that to budget=0 I get 0 clicks
                rewards_per_subcampaign[subcampaign - 1] = gpts_learner[subcampaign - 1].means

            collected_rewards = sum(collected_rewards_per_subcampaign)
            regrets_per_experiment[e].append(optimal_click - collected_rewards)
            budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                            rewards_per_subcampaign=rewards_per_subcampaign)

        for subcampaign in [1, 2, 3]:
            rewards_per_subcampaign_per_experiment[subcampaign - 1][e] = gpts_learner[subcampaign - 1].means

    mean_rewards_per_subcampaign = np.mean(rewards_per_subcampaign_per_experiment, axis=1)
    final_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                          rewards_per_subcampaign=mean_rewards_per_subcampaign)
    budget_indices = [np.max(np.argwhere(list_budgets <= final_budget_allocation[i])) for i in range(3)]
    adv_rew = [mean_rewards_per_subcampaign[i][budget_indices[i]] for i in range(3)]

    if graphics:
        for i in range(3):
            plt.figure()
            if not sliding_window:
                x = np.linspace(np.min(list_budgets), np.max(list_budgets), 100)
                plt.plot(x, n_for_b[i + 1](x), 'r', label='Real function')
                plt.legend()
            plt.plot(list_budgets, mean_rewards_per_subcampaign[i], 'b.')
            plt.xlabel('Budget')
            plt.ylabel('Number of clicks')
            plt.title("Check on the learned curves")
        plt.figure()
        for i in range(3):
            plt.plot(list_budgets, mean_rewards_per_subcampaign[i], '.', label=i+1)
            plt.title("Check on the learned curves")
            plt.legend()

        # Plotting the regret during the learning phase
        plt.figure()
        plt.plot(np.cumsum(np.mean(regrets_per_experiment, axis=0)))
        plt.xlabel('Time')
        plt.ylabel('Number of clicks lost')
        plt.title("Cumulative regret during the first phase")
        plt.show()

    if verbose:
        print("The budget is split as follow: ", final_budget_allocation)
        print("Expected clicks with the optimal budget allocation: ", adv_rew)

    return adv_rew

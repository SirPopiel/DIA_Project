import numpy as np
import matplotlib.pyplot as plt
import random
import time
import winsound
from bidding_environment import *
from moving_bidding_environment import *
from ts_learner_gauss import *
# from swts_learner_gauss import *
from optimization import dynamic_opt


def t_to_phase(subcampaign, time_horizon, t_):
    """Returns the phase from the current time."""
    
    for p in range(3):
        if n_proportion_phases[subcampaign][p] * time_horizon >= t_:
            return p
    return 0


def budget_optimizer_ts(budget, list_budgets, sigma, time_horizon, n_tuning=1000, n_experiments=1,
                        sliding_window=False, window_size=0, abrupt_phases=None, graphics=False, verbose=False):
    n_arms = len(list_budgets)
    rewards_per_subcampaign_per_experiment = [[[] for _ in range(n_experiments)] for _ in range(3)]
    regrets_per_experiment = [[] for _ in range(n_experiments)]
    budget_index = np.max(np.argwhere(list_budgets <= budget))

    # Computing the optimal budget allocation in the case of stationary environment
    if not sliding_window:
        real_rewards = [[] for _ in range(3)]
        for subcampaign in [1, 2, 3]:
            real_rewards[subcampaign - 1] = n_for_b[subcampaign](list_budgets)
        optimal_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                                rewards_per_subcampaign=real_rewards)
        optimal_click = sum([n_for_b[i + 1](optimal_budget_allocation[i]) for i in range(3)])
        if verbose:
            print("The best budget allocation would have been: \n", optimal_budget_allocation)
            print("with corresponding expected number of clicks: \n",
                  [n_for_b[i + 1](optimal_budget_allocation[i]) for i in range(3)], "\n")

    for e in range(n_experiments):
        start_time_experiment = time.time()
        envs = []
        collected_rewards_per_subcampaign = np.zeros(3)
        rewards_per_subcampaign = [[] for _ in range(3)]
        ts_learner = []

        for subcampaign in [1, 2, 3]:
            if sliding_window:
                envs.append(MovingBiddingEnvironment(budgets=list_budgets, sigma=sigma, time_horizon=time_horizon,
                                                     subcampaign=subcampaign))
                ts_learner.append(SWTS_Learner(n_arms=n_arms,
                                               window_size=window_size))
            else:
                envs.append(BiddingEnvironment(budgets=list_budgets, sigma=sigma, subcampaign=subcampaign))
                ts_learner.append(TS_Learner_Normal(n_arms=n_arms, arms=list_budgets, sigma=sigma))

        # Initializing the budget allocation evenly
        # budget_allocation = np.max(list_budgets[np.argwhere(list_budgets <= budget / 3)]) * np.ones(3)

        # Initializing the budget allocation randomly
        budget_sub1 = random.choice(list_budgets)
        budget_sub2 = random.choice(list_budgets[np.argwhere(list_budgets <= budget - budget_sub1)])
        budget_sub3 = random.choice(list_budgets[np.argwhere(list_budgets <= budget - budget_sub1 - budget_sub2)])
        budget_allocation = [budget_sub1, budget_sub2[0], budget_sub3[0]]
        # In order to don't give any preference over any subcampaign
        np.random.shuffle(budget_allocation)

        if sliding_window:
            last_phase = 0

        # Starting each experiment
        for t in range(time_horizon):
            # Computing the optimal budget allocation in the case of non-stationary environment
            if sliding_window and last_phase < len(abrupt_phases) and abrupt_phases[last_phase] * time_horizon >= t:
                real_rewards = [[] for _ in range(3)]
                for subcampaign in [1, 2, 3]:
                    phase_sub = t_to_phase(subcampaign, time_horizon, t)
                    real_rewards[subcampaign - 1] = n_for_b[subcampaign][phase_sub](list_budgets)
                optimal_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                                        rewards_per_subcampaign=real_rewards)
                optimal_click = sum([n_for_b[i + 1][phase_sub](optimal_budget_allocation[i]) for i in range(3)])
                last_phase += 1

            # Starting exploration of our environments
            for subcampaign in [1, 2, 3]:
                pulled_arm = ts_learner[subcampaign - 1].pull_arm(budget_allocation[subcampaign - 1])
                reward = envs[subcampaign - 1].round(pulled_arm)
                collected_rewards_per_subcampaign[subcampaign - 1] = reward
                ts_learner[subcampaign - 1].update(pulled_arm, reward)
                # Already know that to budget=0 I get 0 clicks
                ts_learner[subcampaign - 1].gaussian_parameters[0, 0] = 0
                rewards_per_subcampaign[subcampaign - 1] = ts_learner[subcampaign - 1].gaussian_parameters[:, 0]

            collected_rewards = sum(collected_rewards_per_subcampaign)
            regrets_per_experiment[e].append(optimal_click - collected_rewards)
            budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                            rewards_per_subcampaign=rewards_per_subcampaign)

        for subcampaign in [1, 2, 3]:
            rewards_per_subcampaign_per_experiment[subcampaign - 1][e] = ts_learner[subcampaign - 1].gaussian_parameters[:, 0]

        print("Experiment " + str(e + 1) + " of " + str(n_experiments) +
              "--- %.2f seconds." % (time.time() - start_time_experiment))

    mean_rewards_per_subcampaign = np.mean(rewards_per_subcampaign_per_experiment, axis=1)
    final_budget_allocation = dynamic_opt(budget_list=list_budgets, budget_index=budget_index,
                                          rewards_per_subcampaign=mean_rewards_per_subcampaign)
    budget_indices = [np.max(np.argwhere(list_budgets <= final_budget_allocation[i])) for i in range(3)]
    adv_rew = [mean_rewards_per_subcampaign[i][budget_indices[i]] for i in range(3)]

    if graphics:
        for i in range(3):
            fig = plt.figure()
            if not sliding_window:
                x = np.linspace(np.min(list_budgets), np.max(list_budgets), 100)
                plt.plot(x, n_for_b[i + 1](x), 'r', label='Real function')
            plt.plot(list_budgets, mean_rewards_per_subcampaign[i], 'b.', label=i + 1)
            plt.xlabel('Budget')
            plt.ylabel('Number of clicks')
            plt.title("Check on the learned curves")
            plt.legend()
            if sliding_window:
                fig.savefig('Output/Pictures' + 'Non-stationary' + 'Learned_curve_ts' + str(i + 1) + '.png')
            else:
                fig.savefig('Output/Pictures' + 'Stationary' + 'Learned_curve_ts' + str(i + 1) + '.png')

        plt.figure()
        for i in range(3):
            plt.plot(list_budgets, mean_rewards_per_subcampaign[i], '.', label=i + 1)
            plt.title("Check on the learned curves")
            plt.legend()

        # Plotting the regret during the learning phase
        fig = plt.figure()
        plt.plot(np.cumsum(np.mean(regrets_per_experiment, axis=0)))
        plt.xlabel('Time')
        plt.ylabel('Number of clicks lost')
        plt.title("Cumulative regret during the first phase")
        if sliding_window:
            fig.savefig('Output/Pictures' + 'Non-stationary' + 'Cumulative_regret_ts.png')
        else:
            fig.savefig('Output/Pictures' + 'Stationary' + 'Cumulative_regret_ts.png')

    if verbose:
        print("The budget is split as follow: \n", final_budget_allocation)
        print("Expected clicks with the optimal budget allocation: \n", adv_rew)

    # if sliding_window:
    #     f = open("Output/Non-stationary.txt", "w")
    # else:
    #     f = open("Output/Stationary.txt", "w")
    # f.write("Results obtained with " + str(n_experiments) + " and time horizon equal to " + str(time_horizon) + "\n\n")
    #
    # if not sliding_window:
    #     f.write("The best budget allocation would have been: " + str(optimal_budget_allocation) + "\n")
    #     f.write("with corresponding number of clicks: " +
    #             str([n_for_b[i + 1](optimal_budget_allocation[i]) for i in range(3)]) + "\n")
    # f.write("The budget is split as follow: " + str(final_budget_allocation) + "\n")
    # f.write("Expected clicks with the optimal budget allocation: " + str(adv_rew) + "\n")
    # f.close()

    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

    return adv_rew

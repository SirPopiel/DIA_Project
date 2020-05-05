import numpy as np
import matplotlib.pyplot as plt
import pulp
from bidding_environment import *
from MovingBiddingEnvironment import *
from gpts_learner import *
from SlidingWindowsGPTS_Learner import *



def bid_optimizer(bids, n_arms, sigma, time_horizon, sliding_window=False, window_size=0, verbose=False, graphics=False):
    regrets_per_subcampaign = []
    rewards_per_subcampaign = []
#    means_subcampaign = []

    for subcampaign in [1, 2, 3]:
        if sliding_window:
            env = MovingBiddingEnvironment(bids=bids, sigma=sigma, time_horizon=time_horizon, subcampaign=subcampaign)
            gpts_learner = SlidingWindowsGPTS_Learner(n_arms=n_arms, arms=bids, window_size=window_size)
            gpts_learner.sigmas[0] = 0
            regrets_per_subcampaign.append([])
        else:
            env = BiddingEnvironment(bids=bids, sigma=sigma, subcampaign=subcampaign)
            gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)

        for t in range(time_horizon):
            pulled_arm = gpts_learner.pull_arm()
            reward = env.round(pulled_arm)
            gpts_learner.update(pulled_arm, reward)
            if sliding_window:
                regrets_per_subcampaign[subcampaign - 1].append(np.max(env.means) - reward)

        print(gpts_learner.means)
        print(gpts_learner.sigmas)
        #todo: Prendiamo questo come reward o semplicemente i collected_rewards?
        rewards_per_subcampaign.append(gpts_learner.means - gpts_learner.sigmas)
#        rewards_per_subcampaign.append(gpts_learner.collected_rewards)
        #        rewards_per_sgpts_rewards_per_experiment.append(gpts_learner.collected_rewards)
        if not sliding_window:
            opt = np.max(env.means)  # todo:check this
            regrets_per_subcampaign.append(opt - gpts_learner.collected_rewards)

        # Save the means learned during the for cicle
#        means_subcampaign.append(gpts_learner.means)

    def get_reward(i, sub):
        return rewards_per_subcampaign[sub - 1][i]

    def get_bid(i):
        return bids[i]

    sub_1_choice = pulp.LpVariable.dicts('sub_1_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    sub_2_choice = pulp.LpVariable.dicts('sub_2_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    sub_3_choice = pulp.LpVariable.dicts('sub_3_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    p1_model = pulp.LpProblem("Bid_Optimization_Model", pulp.LpMaximize)
    p1_model += (
            sum([get_reward(choice, 1) * sub_1_choice[choice] for choice in range(n_arms)]) +
            sum([get_reward(choice, 2) * sub_2_choice[choice] for choice in range(n_arms)]) +
            sum([get_reward(choice, 3) * sub_3_choice[choice] for choice in range(n_arms)])
    )

    p1_model += sum([sub_1_choice[choice] for choice in range(n_arms)]) <= 1
    p1_model += sum([sub_2_choice[choice] for choice in range(n_arms)]) <= 1
    p1_model += sum([sub_3_choice[choice] for choice in range(n_arms)]) <= 1

    p1_model += (
                        sum([get_bid(choice) * sub_1_choice[choice] for choice in range(n_arms)]) +
                        sum([get_bid(choice) * sub_2_choice[choice] for choice in range(n_arms)]) +
                        sum([get_bid(choice) * sub_3_choice[choice] for choice in range(n_arms)])
                ) <= 1.0

    p1_model.solve()

    adv_rew = [[] for _ in range(3)]

    for choice in range(n_arms):
        if sub_1_choice[choice].value() == 1.0:
            if verbose:
                print(1, choice, bids[choice], get_reward(choice, 1))
            adv_rew[0] = get_reward(choice, 1)
        if sub_2_choice[choice].value() == 1.0:
            if verbose:
                print(2, choice, bids[choice], get_reward(choice, 2))
            adv_rew[1] = get_reward(choice, 2)
        if sub_3_choice[choice].value() == 1.0:
            if verbose:
                print(3, choice, bids[choice], get_reward(choice, 3))
            adv_rew[2] = get_reward(choice, 3)

    if graphics:
        plt.figure(0)
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(regrets_per_subcampaign[0]), 'g')
        plt.plot(np.cumsum(regrets_per_subcampaign[1]), 'b')
        plt.plot(np.cumsum(regrets_per_subcampaign[2]), 'r')
        plt.legend(["GPTS_1", "GPTS_2", "GPTS_3"])
        plt.show()

    '''
    Ci serve parte intera dei click o no?
    '''
    return adv_rew
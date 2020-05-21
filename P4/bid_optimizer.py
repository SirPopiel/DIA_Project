import numpy as np
import matplotlib.pyplot as plt
import pulp
from bidding_environment import *
from MovingBiddingEnvironment import *
from gpts_learner import *
from SlidingWindowsGPTS_Learner import *
from scipy import interpolate
from optimization import *
import time


def bid_optimizer(bids, n_arms, sigma, time_horizon, sliding_window=False,
                  window_size=0, verbose=False, graphics=False):
    regrets_per_subcampaign = []
    rewards_per_subcampaign = []
    gpts_learner = []

    for subcampaign in [1, 2, 3]:
        if sliding_window:
            env = MovingBiddingEnvironment(bids=bids, sigma=sigma, time_horizon=time_horizon, subcampaign=subcampaign)
            gpts_learner.append(SlidingWindowsGPTS_Learner(n_arms=n_arms, arms=bids, window_size=window_size))
            regrets_per_subcampaign.append([])
        else:
            env = BiddingEnvironment(bids=bids, sigma=sigma, subcampaign=subcampaign)
            gpts_learner.append(GPTS_Learner(n_arms=n_arms, arms=bids))

        for t in range(time_horizon):
            pulled_arm = gpts_learner[subcampaign - 1].pull_arm()
            reward = env.round(pulled_arm)
            gpts_learner[subcampaign - 1].update(pulled_arm, reward)
            if sliding_window:
                #regrets_per_subcampaign[subcampaign - 1].append(np.max(env.means) - reward)
                regrets_per_subcampaign[subcampaign - 1].append(np.max(env.means) - env.means[pulled_arm])
        '''
        Il regret va calcolato così, considerando la media, non le fluttuazioni stocastiche
        '''

        '''
        We discard low accuracy values of gpts_learner.means because they weren't pulled enough and they
        would be dragged by higher bids to non realistic values. Moreover, we know that with bid = 0.0
        the number of clicks has to be zero. So we interpolate between 0 and the first "accurate" value
        we found. We consider "accurate" the first value with std. dev. lower than the median of the std
        devs computed
        '''
        idx_accurate = np.argwhere(gpts_learner[subcampaign - 1].sigmas
                                   < np.median(gpts_learner[subcampaign - 1].sigmas))[0][0]
        # todo: trovare un modo più elegante di questo
        temporary_rewards = gpts_learner[subcampaign - 1].means    # todo: vogliamo sottrarre la dev. std?
        x = [0, bids[idx_accurate], bids[idx_accurate + 1]]     #todo: questo idx_accurate + 1 teoricamente dà problemi
        y = [0, temporary_rewards[idx_accurate], temporary_rewards[idx_accurate + 1]]
        tck = interpolate.splrep(x, y, k=2)
        xnew = np.linspace(0, bids[idx_accurate], idx_accurate)
        ynew = interpolate.splev(xnew, tck)
        temporary_rewards[0:idx_accurate] = ynew

        rewards_per_subcampaign.append(temporary_rewards)

        ### DA MODIFICARE, FUNZIONAVA SOLO CON MAX BID = 1
        if not sliding_window:
            opt = np.max(env.means)  # todo:check this
            mean_rewards = []
            for p in gpts_learner[subcampaign - 1].pulled_arms:
                mean_rewards.append(env.means[int(p*(n_arms-1))])
            regrets_per_subcampaign.append(opt - mean_rewards)
            #regrets_per_subcampaign.append(opt - gpts_learner[subcampaign - 1].collected_rewards)

        '''
        così dovrebbe essere corretto, seppur non efficiente, la curva è perlomeno monotona
        '''


    if graphics:
        x_pred = np.atleast_2d(bids).T
        for i in range(3):
            y_pred, sigma = gpts_learner[i].gp.predict(x_pred, return_std=True)
            #for hyperparameter in gpts_learner[i].gp.kernel.hyperparameters:
                 #print("Hyperparameters : ", hyperparameter)
            plt.figure()
            if sliding_window:
                plt.plot(bids, rewards_per_subcampaign[i], 'o', label='Quadratic interpolation in low bids')
            else:
                x = np.linspace(np.min(bids), np.max(bids), 100)
                plt.plot(x, n_to_f[i + 1](x), 'r', label='Real function')
                plt.plot(bids, rewards_per_subcampaign[i], 'o', label='Quadratic interpolation in low bids')
                plt.legend(["Real function", "Quadratic interpolation in low bids"])
            plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
            plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                     np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                     alpha=.5, fc='b', ec='None', label='95% Confidence Interval')
            plt.xlabel('$x$')
            plt.ylabel('$n(x)$')
            plt.title("Check on the learned curves")
            plt.legend(loc='lower right')


    def get_reward(i, sub):
        return rewards_per_subcampaign[sub - 1][i]

    def get_bid(i):
        return bids[i]

    start_time = time.time()
    subs = dynamic_opt(range(n_arms),n_arms-1,rewards_per_subcampaign,1)
    elapsed_time = time.time() - start_time
    print("Dynamic_time = ", elapsed_time)

    adv_rew = [0 for _ in range(3)]

    """
    start_time = time.time()
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

    p1_model.writeLP(filename = "allocation.lp")
    p1_model.solve()

    elapsed_time = time.time() - start_time
    print("LP_time = ", elapsed_time)

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

    """
    for i in range(len(subs)):
        adv_rew[i] = get_reward(subs[i], i+1)






    if graphics:
        plt.figure()
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(regrets_per_subcampaign[0]), 'g')
        plt.plot(np.cumsum(regrets_per_subcampaign[1]), 'b')
        plt.plot(np.cumsum(regrets_per_subcampaign[2]), 'r')
        plt.legend(["GPTS_1", "GPTS_2", "GPTS_3"])

    cumulative_regret = []
    for i in range(3):
        cumulative_regret.append(sum(cumulative_regret)*np.ones(time_horizon)
                                 + regrets_per_subcampaign[i])
    if graphics:
        plt.figure()
        plt.xlabel("t")
        plt.ylabel("Regret")
        plt.plot(np.cumsum(cumulative_regret), label='Cumulative regret learning all curves')
        plt.legend()
        plt.show()


    # todo: leggi qui
    '''
    Ci serve parte intera dei click o no?
    '''
    return adv_rew

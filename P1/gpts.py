import numpy as np
import matplotlib.pyplot as plt
from bidding_environment import *
from gts_learner import *
from gpts_learner import *
import pulp

n_arms = 25
min_bid = 0.0
max_bid = 1.0
bids = np.linspace(min_bid, max_bid, n_arms)
sigma = 10

T = 150

regrets_per_subcampaign = []
rewards_per_subcampaign = []

for subcampaing in [1, 2, 3]:
    env = BiddingEnvironment(bids=bids, sigma=sigma, subcampaing=subcampaing)
    gpts_learner = GPTS_Learner(n_arms=n_arms, arms=bids)
    for t in range(T):
        pulled_arm = gpts_learner.pull_arm()
        reward = env.round(pulled_arm)
        gpts_learner.update(pulled_arm, reward)

    rewards_per_subcampaign.append(gpts_learner.means - gpts_learner.sigmas)
    opt = np.max(env.means) #todo:check this
    regrets_per_subcampaign.append(opt - gpts_learner.collected_rewards)

# print(rewards_per_subcampaign)
def get_reward(i, sub):
    return rewards_per_subcampaign[sub - 1][i]

def get_bid(i):
    return bids[i]

sub_1_choice = pulp.LpVariable.dicts('sub_1_choice', [i for i in range(n_arms)], 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)

sub_2_choice = pulp.LpVariable.dicts('sub_2_choice', [i for i in range(n_arms)], 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)

sub_3_choice = pulp.LpVariable.dicts('sub_3_choice', [i for i in range(n_arms)], 
                            lowBound = 0,
                            upBound = 1,
                            cat = pulp.LpInteger)

p1_model = pulp.LpProblem("P1 Model", pulp.LpMaximize)
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

for choice in range(n_arms):
    if sub_1_choice[choice].value() == 1.0:
        print(1, choice, bids[choice], get_reward(choice, 1))
    if sub_2_choice[choice].value() == 1.0:
        print(2, choice, bids[choice], get_reward(choice, 2))
    if sub_3_choice[choice].value() == 1.0:
        print(3, choice, bids[choice], get_reward(choice, 3))



# plt.figure(0)
# plt.xlabel("t")
# plt.ylabel("Regret")
# plt.plot(np.cumsum(np.mean(opt-gpts_rewards_per_experiment, axis=0)), 'g')
# plt.legend(["GPTS"])
# plt.show()

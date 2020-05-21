from PricingEnvironment import *
from TSLearner import *
from ContextGenerator import *
import math
import numpy as np

memory_loss = False
# TIME HORIZON AND CAMPAIGNS
T = 119
n_campaigns = 3
# CLICKS OBTAINED BY ADVERTISING
clicks = [1000, 800, 1000]
n_users = sum(clicks)

# PRICES AND THEIR DISCRETIZATION
n_arms = 20
price_min = 100
price_max = 200
delta_price = price_max - price_min
prices = np.linspace(price_min, price_max, n_arms)
# CONVERSION RATE CURVES
p = {
    0: (lambda x: 27*(6*((x-price_min)*2/delta_price) - ((x-price_min)*2/delta_price)**2 - ((x-price_min)*2/delta_price)**3)/(38*math.sqrt(19)-56)*0.65),
    1: (lambda x: 27*(6*((x-price_min)*2/delta_price) - ((x-price_min)*2/delta_price)**2 - ((x-price_min)*2/delta_price)**3)/(38*math.sqrt(19)-56)*0.65),
    #2: (lambda x: 27*(6*((x-price_min)*2/delta_price) - ((x-price_min)*2/delta_price)**2 - ((x-price_min)*2/delta_price)**3)/(38*math.sqrt(19)-56)*0.30)
    2: (lambda x: 0.8 - 0.3*(x - price_min)/delta_price)
}
'''
first_learner = TS_Learner(n_arms = n_arms, n_users = n_users)
print(first_learner.beta_parameters)
cg_env = PricingEnvironment(campaigns = init_campaigns, clicks = clicks, prices = prices, probabilities = p )
'''
# CONTEXT INITIALIZED AS AGGREGATED, WITH RELATIVE ENVIRONMENT AND LEARNER
contexts = [[0,1,2]]
environments = []
learners = []
environments.append(PricingEnvironment(contexts[0], clicks, prices, p))
learners.append(TS_Learner(contexts[0],n_arms, n_users))
# CONTEXT GENERATOR
#generator = ContextGenerator(clicks, confidence = 0.95)
#
for t in range(T):
### RUN A BANDIT FOR EVERY CONTEXT #############################################
    for cid in range(len(contexts)):
        c = contexts[cid]
        pulled_arm = learners[cid].pull_arm()
        rewards = environments[cid].round(pulled_arm)
        '''
        if sum(rewards):
            context_p = [i/sum(rewards) for i in rewards]
        else:
            context_p = [0 for _ in range(len(rewards))]
        context_reward = sum([rewards[j] for j in c])
        '''
        print('context {c}: arm {a}'.format(c = c, a = pulled_arm))
        learners[cid].update(pulled_arm, rewards)
        #print(pulled_arm,learners[cid].rewards_per_arm)

#### WEEKEND ###################################################################
    if not (t + 1) % 7 and t and len(contexts) < 3:
        #print(sum(learners[cid].collected_rewards))
        #print(learners[cid].context_p())
        todel = []
        for cid in range(len(contexts)):
            c = contexts[cid]
            if len(c) > 1:
                best_arms, best_arms_rew, best_arm_full, best_arm_rew_full = learners[cid].opt_arm_reward()
                context_probabilities = learners[cid].context_p()
                is_split, new_contexts = split(c, best_arms_rew, best_arm_rew_full, context_probabilities)
                if not is_split:
                    if memory_loss:
                        learners[cid].rewards_per_arm = [[] for i in range(n_arms)]
                        learners[cid].collected_rewards = np.array([])
                else:
                    todel.append(cid)
                    #print(todel)
                    for c in new_contexts:
                        cclicks = [clicks[cid] for cid in c]
                        cp = [p[cid] for cid in c]
                        environments.append(PricingEnvironment(c, cclicks, prices, cp))
                        learners.append(TS_Learner(c, n_arms, n_users))
                    print('Splitted context')
                #print(new_contexts)

        contexts += new_contexts
        #print(contexts)
        for d in todel:
            del contexts[d]
            del learners[d]
            del environments[d]
        #print(contexts)
################################################################################

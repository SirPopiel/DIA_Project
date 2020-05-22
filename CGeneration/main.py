from PricingEnvironment import *
from TSLearner import *
from ContextGenerator import split
import math
import numpy as np
# MUST THE LEARNER FORGOT INFORMATIONS AFTER ONE WEEK PASSED? NO
memory_loss = False
# NUMBER OF EXPERIMENTS
n_experiments = 50
# TIME HORIZON AND NUMBER OF CAMPAIGNS
T = 119
n_campaigns = 3
# CLICKS OBTAINED BY ADVERTISING
clicks = [800, 800, 1200]
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
# OPTIMAL PROBABILITIES FOR EACH SUBCAMPAIGN
opt = []
for i in range(3):
    opt.append(max([p[i](arm) for arm in prices]))
# REGRET INITIALIZED AS EMPTY
regret = [[] for _ in range(3)]
for e in range(n_experiments):
    if not e % 10:
        print(e)
### CONTEXT INITIALIZED AS AGGREGATED, WITH RELATIVE ENVIRONMENT AND LEARNER ################################
    rewards_per_experiment = []
    contexts = [[0,1,2]]
    environments = []
    learners = []
    environments.append(PricingEnvironment(contexts[0], clicks, prices, p))
    learners.append(TS_Learner(contexts[0],n_arms, n_users))
    for t in range(T):
####### RUN A BANDIT FOR EVERY CONTEXT ######################################################################
        for cid in range(len(contexts)):
            c = contexts[cid]
            pulled_arm = learners[cid].pull_arm()
            rewards = environments[cid].round(pulled_arm)
            learners[cid].update(pulled_arm, rewards)
########### FOR EVERY SUBCAMPAIGN IN THE CONTEXT WE COMPUTE THE REGRET #######################################
            for sc in range(len(c)):
                nit = range(sc,len(learners[cid].collected_rewards),len(c))
                cr = [opt -learners[cid].collected_rewards[cr]/clicks[c[sc]] for cr in nit]
                regret[c[sc]].append(np.mean(cr))
####### WEEKEND ##############################################################################################
        if not (t + 1) % 7 and t and len(contexts) < 3:
            todel = [] #  TO STORE EVENTUAL CONTEXT TO BE DELETED AFTER SPLIT
########### I DO THE SPLIT CHECK FOR EVERY CONTEXT SPLITTABLE (> 1 SUBCAMPAIGN) ##############################
            for cid in range(len(contexts)):
                c = contexts[cid]
                if len(c) > 1:
################### RETRIEVE SUBCAMPAIGN PROBABILITIES AND EXP. VALUES FROM LEARNER DATAS #####################
                    best_arms, best_arms_rew, best_arm_full, best_arm_rew_full = learners[cid].opt_arm_reward()
                    context_probabilities = learners[cid].context_p()
################### SPLIT CHECK AND EVENTUAL NEW CONTEXTS #####################################################
                    is_split, new_contexts = split(c, best_arms_rew, best_arm_rew_full, context_probabilities)
                    if not is_split: # WE DON'T SPLIT, CONTEXT REMAINS THE SAME
                        if memory_loss: #INFAMOUS MEMORY LOSS, SUGGEST NOT TO
                            learners[cid].rewards_per_arm = [[] for i in range(n_arms)]
                            learners[cid].collected_rewards = np.array([])
################### WE SPLIT, OLD CONTEXT TO DELETE AND NEW ENVIRONMENTS AND LEARNERS TO INITIALIZE FOR NEW CONTEXT
                    else:
                        todel.append(cid) # AFTER SPLIT WE NEED TO REMOVE OLD LEARNER AND RELATIVE ENVIRONMENT
                        for c in new_contexts:
                            cclicks = [clicks[cid] for cid in c]
                            cp = [p[cid] for cid in c]
                            environments.append(PricingEnvironment(c, cclicks, prices, cp))
                            learners.append(TS_Learner(c, n_arms, n_users))
                        print('Splitted context')

            contexts += new_contexts # WE UPDATE ACTUAL CONTEXTS ADDING THE NEW ONES
            for d in todel:
                del contexts[d] # WE DELETE OLD AGGREGATED CONTEXT
                del learners[d] # WE DELETE OLD LEARNER
                del environments[d] # WE DELETE OLD ENVIRONMENT
################################################################################
print(regret)
print(np.cumsum(regret))

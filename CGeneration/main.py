from PricingEnvironment import *
from TSLearner import *
from ContextGenerator import split
import math
import numpy as np
import matplotlib.pyplot as plt
# MUST THE LEARNER FORGET INFORMATIONS AFTER ONE WEEK PASSED? NO
memory_loss = False
# Split allowed 
# False -> P4
# True -> P5
split_allowed = True
# MUST THE CONTEXT PROBABILITIES BE STATIC?
static_prob = False
# NUMBER OF EXPERIMENTS
n_experiments = 5
# TIME HORIZON AND NUMBER OF CAMPAIGNS
T = 150
n_campaigns = 3
# CLICKS OBTAINED BY ADVERTISING
clicks = [40, 37, 77]
n_users = sum(clicks)
# PRICES AND THEIR DISCRETIZATION
n_arms = 20
price_min = 250
price_max = 500
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
    opt.append(max([p[i](arm)*arm*clicks[i] for arm in prices]))
# REGRET INITIALIZED AS EMPTY
all_rewards = [[[] for _ in range(3)] for _ in range(n_experiments)]
split_times = []
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
            learners[cid].update(pulled_arm, rewards, prices[pulled_arm])
########### FOR EVERY SUBCAMPAIGN IN THE CONTEXT WE COMPUTE THE REGRET #######################################
            for sc in range(len(c)):
                nit = range(sc,len(learners[cid].collected_rewards),len(c))
                all_rewards[e][c[sc]].append([learners[cid].collected_rewards[r]/clicks[c[sc]] for r in nit][-1])
            #print(learners[cid].collected_rewards)
                # cr = [(opt -learners[cid].collected_rewards[r])/clicks[c[sc]] for r in nit]
                # regrets[e][c[sc]].append(cr)
            print('context : {}       pulled_arm : {}         reward : {}'.format(c,pulled_arm,[reward*prices[pulled_arm] for reward in rewards]))
####### WEEKEND ##############################################################################################
        if not (t + 1) % 7 and t and len(contexts) < 3 and split_allowed:
            todel = [] #  TO STORE EVENTUAL CONTEXT TO BE DELETED AFTER SPLIT
########### I DO THE SPLIT CHECK FOR EVERY CONTEXT SPLITTABLE (> 1 SUBCAMPAIGN) ##############################
            for cid in range(len(contexts)):
                c = contexts[cid]
                if len(c) > 1:
################### RETRIEVE SUBCAMPAIGN PROBABILITIES AND EXP. VALUES FROM LEARNER DATAS #####################
                    best_arms, best_arms_rew, best_arm_full, best_arm_rew_full = learners[cid].opt_arm_reward()
                    if not static_prob:
                        context_probabilities = learners[cid].context_p()
                    else:
                        cclicks= [clicks[sc] for sc in c]
                        context_probabilities = [cclicks[sc]/sum(cclicks) for sc in c]
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
                            learners.append(TS_Learner(c, n_arms, sum(cclicks)))
                        print('Splitted context')
                        #if not t in split_times:
                        split_times.append(t)

            contexts += new_contexts # WE UPDATE ACTUAL CONTEXTS ADDING THE NEW ONES
            for d in todel:
                del contexts[d] # WE DELETE OLD AGGREGATED CONTEXT
                del learners[d] # WE DELETE OLD LEARNER
                del environments[d] # WE DELETE OLD ENVIRONMENT
################################################################################
#print(regret)
#print(np.cumsum(regret))

# Note: we collect and print all the splits that happen across all n_experiments
unique_split_times = list(set(split_times))
print(unique_split_times)
#print(all_rewards) # experiment subcampaign time
# mean on experiments
mean_rewards = [[np.mean([all_rewards[iexp][isc][itime] for iexp in range(n_experiments)]) for itime in range(T)] for isc in range(3)]
mean_regrets = [[opt[isc]/clicks[isc] - mean_rewards[isc][itime] for itime in range(T)] for isc in range(3)]

plt.figure(0)
plt.ylabel("Rewards")
plt.xlabel("t")
plt.plot(range(T),mean_rewards[0], 'r')
plt.plot(range(T),mean_rewards[1], 'g')
plt.plot(range(T),mean_rewards[2], 'b')
plt.title("Rewards learning the best price")
for st in unique_split_times:
    plt.axvline(st)

plt.figure(1)
plt.ylabel("Regrets")
plt.xlabel("t")
plt.plot(range(T),mean_regrets[0], 'r')
plt.plot(range(T),mean_regrets[1], 'g')
plt.plot(range(T),mean_regrets[2], 'b')
plt.title("Regrets learning the best price")
for st in unique_split_times:
    plt.axvline(st)

total_regrets = [sum([mean_regrets[isc][imr]*clicks[isc] for isc in range(3)])/n_users for imr in range(T)]
plt.figure(2)
plt.ylabel("Total Regrets")
plt.xlabel("t")
plt.plot(range(T), total_regrets, 'b')
plt.title("Total Regrets learning the best price")
for st in unique_split_times:
    plt.axvline(st)

plt.figure(3)
plt.ylabel("Cumulative Regrets")
plt.xlabel("t")
plt.plot(range(T),np.cumsum(mean_regrets[0]), 'r')
plt.plot(range(T),np.cumsum(mean_regrets[1]), 'g')
plt.plot(range(T),np.cumsum(mean_regrets[2]), 'b')
plt.title("Cumulative Regrets learning the best price")
for st in unique_split_times:
    plt.axvline(st)

plt.figure(4)
plt.ylabel("Total Cumulative Regret")
plt.xlabel("t")
plt.plot(range(T), np.cumsum(total_regrets), 'b')
plt.title("Total Cumulative Regret learning the best price")
for st in unique_split_times:
    plt.axvline(st)

plt.show()
from utilities import *
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
################################### PARAMETERS #################################
split_allowed = True
n_experiments = 10
T = 300
n_arms = 20
price_min = 50
price_max = 70
clicks = {1 : 50, 2 : 50, 3 : 50}
n_users = sum(clicks.values())
delta_price = price_max - price_min
prices = np.linspace(price_min, price_max, n_arms)
conversion_rate = {
	1: (lambda x: 27*(6*((x-price_min)*2/delta_price) - ((x-price_min)*2/delta_price)**2 - ((x-price_min)*2/delta_price)**3)/(38*math.sqrt(19)-56)*0.65),
	#1: (lambda x: (((x-price_min)/delta_price)**2 + 3*((x-price_min)/delta_price) - 4*(((x-price_min)/delta_price)**4))/2),
	#1: (lambda x: 27*(6*((x-price_min)*2/delta_price) - ((x-price_min)*2/delta_price)**2 - ((x-price_min)*2/delta_price)**3)/(38*math.sqrt(19)-56)*0.65),
	2: (lambda x: 0.6 - 0.3*(x - price_min)/delta_price),
	3: (lambda x: 0.8+math.sin((x-price_min)/(math.pi*delta_price))**2)
}
context_probabilities = {sc : clicks[sc]/n_users for sc in [1,2,3]}
################################################################################

all_rewards = {e : {1 : 0, 2 : 0, 3 : 0} for e in range(n_experiments)}

for e in range(n_experiments):
    letter = 'A'
    contexts = {'A' : [1,2,3]}
    environments = dict()
    learners = dict()
    environments.update({'A' : PricingEnvironment([1,2,3], prices, conversion_rate)})
    learners.update({'A' : TS_Learner([1,2,3],n_arms)})
    for t in range(T):
        for c in contexts:
            learners[c].status_update()
            pulled_arm = learners[c].pull_arm()
            for sc in c:
                for nc in clicks[sc]:
                	reward = environments[c].round(pulled_arm, sc) # customer buys -> reward = 1, 0 otherwise
                	learners[c].update(pulled_arm, reward, prices[pulled_arm], sc)
                all_rewards[e][sc] += learners[c].collected_rewards[t][sc]
########################## WEEKEND #############################################
            if (t+1)%7 == 0 and t and len(contexts) > 1 and split_allowed:
                best, best_total = learners[c].best_performance()
                split_output = is_split(c, best, best_total, context_probabilities)
				split_bool = split_output[0]
				new_context = split_output[1]
                if split_bool:

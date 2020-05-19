import numpy as np
import random
import math

class PricingEnvironment() :
    def __init__(self, campaigns =  [1,2,3], clicks = [100,100,100]) :
        self.campaigns = campaigns
        self.clients = clicks
        self.probabilities = { ### Conversion rate curves for the various subcampaigns
            1: (lambda x : (0.8*math.sin(math.pi*x/100) + 0.1)),
            2: (lambda x : (0.7*math.sin(math.pi*x/100) + 0.2)),
            3: (lambda x : (0.5*math.sin(math.pi*x/100) + 0.3))
        }
    ### Given the pulled arm returns the number of clients whose have bought from the various subcampaigns
    def round(self, pulled_arm) :
        reward = {'sc'+str(self.campaigns[i]) :
                           np.random.binomial(self.clients[self.campaigns[i]-1], self.probabilities[self.campaigns[i]](pulled_arm)) \
                           for i in range(len(self.campaigns))}
        return reward


'''
clicks = [70,10,70]
env = PricingEnvironment(campaigns = [1,3], clicks = clicks)
rew = env.round(9)
print(rew)
'''

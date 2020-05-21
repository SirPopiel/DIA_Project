import numpy as np
import random
import math

class PricingEnvironment() :
    def __init__(self, context, clicks, prices, probabilities) :
        self.context = context
        self.clicks = clicks
        self.prices = prices
        self.probabilities = probabilities
    ### Given the pulled arm returns the number of clients whose have bought from the various subcampaigns
    def round(self, pulled_arm) :
        reward = [np.random.binomial(self.clicks[i], self.probabilities[i](self.prices[pulled_arm])) for i in range(len(self.context))]
        #print(reward)
        return reward

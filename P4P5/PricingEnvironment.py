import numpy as np
import random

class PricingEnvironment() :
    def __init__(self, context, prices, probabilities) :
        self.context = context
        self.prices = prices
        self.probabilities = probabilities
    ### Given the pulled arm returns the number of clients whose have bought from the various subcampaigns
    def round(self, pulled_arm, sc) :
        reward = np.random.binomial(1, self.probabilities[sc](self.prices[pulled_arm]))
        return reward

import numpy as np
import matplotlib.pyplot as plt
import math

# n_arms = 25

p = {
    1: (lambda x : (0.7*np.exp(-(x-50)**(1/2)/20))),
    2: (lambda x : (0.9*np.exp(-(x-50)**(1/2)/20))),
    3: (lambda x : (0.5*np.exp(-(x-50)**(1/2)/20)))
}

class PricingEnvironment():
    def __init__(self, n_arms, prices, subcampaign=1):
        self.n_arms = n_arms
        self.probabilities = p[subcampaign]
        self.prices = prices

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities(self.prices[pulled_arm]))
        return reward

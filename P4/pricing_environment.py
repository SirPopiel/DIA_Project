import numpy as np


class PricingEnvironment:
    def __init__(self, n_arms, prices, p, subcampaign=1):
        self.n_arms = n_arms
        self.probabilities = p[subcampaign]
        self.prices = prices

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities(self.prices[pulled_arm]))
        return reward

import numpy as np


p = {
    1: np.array([0.31, 0.15, 0.14, 0.10]),
    2: np.array([0.36, 0.18, 0.16, 0.06]),
    3: np.array([0.20, 0.15, 0.15, 0.10])
}

class PricingEnvironment() :
    def __init__(self, n_arms, subcampaign = 1) :
        self.n_arms = n_arms
        self.probabilities = p[subcampaign]

    def round(self, pulled_arm) :
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

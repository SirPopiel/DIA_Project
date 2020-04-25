import numpy as np
import matplotlib.pyplot as plt
import math

n_arms = 25

p = {
    1: (lambda x : (0.8*math.sin(math.pi*x/100) + 0.1)),
    2: (lambda x : (0.7*math.sin(math.pi*x/100) + 0.2)),
    3: (lambda x : (0.5*math.sin(math.pi*x/100) + 0.3))
}

class PricingEnvironment() :
    def __init__(self, n_arms, subcampaign = 1) :
        self.n_arms = n_arms
        self.probabilities = p[subcampaign]

    def round(self, pulled_arm) :
        reward = np.random.binomial(1, self.probabilities(pulled_arm))
        return reward

import numpy as np
from data import n_for_b

class BiddingEnvironment():
    def __init__(self, budgets, sigma, subcampaign=1):
        self.budgets = budgets
        self.means = n_for_b[subcampaign](budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

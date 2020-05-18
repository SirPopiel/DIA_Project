import numpy as np


class BiddingEnvironment():
    def __init__(self, bids, sigma, subcampaign=1):
        self.bids = bids
        self.means = n_to_f[subcampaign](bids)
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

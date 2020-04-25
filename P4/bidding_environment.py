import numpy as np

n_to_f = {
    1: (lambda x: 100 * (1.0 - np.exp(-4*x + 3*x**3))),
    2: (lambda x: 75 * (1.0 - np.exp(-3*x + x**2 + 1*x**3))),
    3: (lambda x: 100 * (1.0 - np.exp(-2*x + 1*x**3)))
}


class BiddingEnvironment():
    def __init__(self, bids, sigma, subcampaing=1):
        self.bids = bids
        self.means = n_to_f[subcampaing](bids)
        self.sigmas = np.ones(len(bids)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

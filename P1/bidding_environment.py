import numpy as np

# (Arbitrary) Bidding curves
n_to_f = {
    1: (lambda x : 100 * (1.0 - np.exp(-4*x + 3*x**3))),
    2: (lambda x : 100 * (1.8 - np.exp(-2*x + x**2 + 1*x**3))),
    3: (lambda x : 75 * (1.5 - np.exp(-1*(x+0.1) + 1*(x+0.1)**3)))
}

class BiddingEnvironment():

    # The init of bidding environment based on the imagined bidding curves
    def __init__(self, bids, sigma, subcampaing=1):
        self.bids = bids
        self.means = n_to_f[subcampaing](bids) # Value of the bidding curve
        self.sigmas = np.ones(len(bids))*sigma # Arbitrary std for each bid

    # Gets a random normal value around the mean of the pulled arm
    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

import numpy as np


class MovingBiddingEnvironment:
    def __init__(self, bids, sigma, time_horizon, subcampaign=1):
        self.bids = bids
        self.time_horizon = time_horizon
        self.subcampaign = subcampaign
        self.sigmas = np.ones(len(bids))*sigma

        self.t_ = 0

    def t_to_phase(self):
        for p in range(N):
            if n_proportion_phases[self.subcampaign][p] * self.time_horizon >= self.t_:
                return p
        return 0

    def round(self, pulled_arm):
        self.means = n_t_to_f[self.subcampaign][self.t_to_phase()](self.bids)
        self.t_ += 1
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
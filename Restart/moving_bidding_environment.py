import numpy as np
from data import n_proportion_phases, n_for_b


class MovingBiddingEnvironment:
    def __init__(self, budgets, sigma, time_horizon, subcampaign=1, N=3):
        self.budgets = budgets
        self.time_horizon = time_horizon
        self.subcampaign = subcampaign
        self.sigmas = np.ones(len(budgets))*sigma
        self.N = N          # number of subcampaigns
        self.t_ = 0

    def t_to_phase(self):
        for p in range(self.N):
            if n_proportion_phases[self.subcampaign][p] * self.time_horizon >= self.t_:
                return p
        return 0

    def round(self, pulled_arm):
        self.means = n_for_b[self.subcampaign][self.t_to_phase()](self.budgets)
        self.t_ += 1
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

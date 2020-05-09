import numpy as np

# Functions that assign the number of clicks to a given bid
# They are monotone increasing in [0,1]
n_t_to_f = {
    1: [(lambda x : 100 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x : 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3))),
        (lambda x : 75 * (1.0 - np.exp(-4*x + 1*x**3)))
    ],
    2: [(lambda x : 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3))),
        (lambda x : 75 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x : 100 * (1.0 - np.exp(-4*x + 1*x**3)))
    ],
    3: [(lambda x : 75 * (1.0 - np.exp(-4*x + 1*x**3))),
        (lambda x : 100 * (1.0 - np.exp(-5*x + 2*x**3))),
        (lambda x : 100 * (1.0 - np.exp(-5*x + x**2 + 1*x**3)))
    ]
}

# Proportion of the time horizon in which each phase takes place
n_proportion_phases = {
    1: [0, 0.3, 0.6],
    2: [0, 0.3, 0.5],
    3: [0, 0.2, 0.6]
}

N = 3

class MovingBiddingEnvironment():
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
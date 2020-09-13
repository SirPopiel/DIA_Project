import numpy as np
from data import n_proportion_phases, n_for_b


class MovingBiddingEnvironment:
    '''Moving Bidding Environment Class'''

    def __init__(self, budgets, sigma, time_horizon, subcampaign = 1, N = 3):
        '''Initialize the Moving Bidding Environment Class with a list of budgets for each subcampaign, sigma, the time horizon, the current subcampaign and the number of subcampaigns'''

        # Assignments and Initializations
        self.budgets = budgets
        self.time_horizon = time_horizon
        self.subcampaign = subcampaign
        self.sigmas = np.ones(len(budgets))*sigma # sigmas are initialized at value sigma
        self.N = N # number of subcampaigns
        self.t_ = 0 # current time

    def t_to_phase(self):
        '''Returns the phase from the current time.'''
        for p in range(self.N):
            if n_proportion_phases[self.subcampaign][p] * self.time_horizon >= self.t_:
                return p
        return 0

    def round(self, pulled_arm):
        '''Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std.'''
        self.means = n_for_b[self.subcampaign][self.t_to_phase()](self.budgets) # gets the number of clicks for the given budget
        self.t_ += 1 # updates time
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

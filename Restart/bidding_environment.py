import numpy as np


class BiddingEnvironment:
    """Bidding Environment Class"""

    def __init__(self, budgets, sigma, subcampaign = 1):
        """Initialize the Bidding Environment Class with a list of budgets for each subcampaign, sigma and the current subcampaign"""

        # Assignments and Initializations
        self.budgets = budgets
        with open('n_for_b.pkl', 'rb') as f:
            n_for_b = pickle.load(f)
        self.means = n_for_b[subcampaign](budgets) # the means are evaluated through a function that assigns the number of clicks to a given budget
        self.sigmas = np.ones(len(budgets)) * sigma # sigmas are initialized at value sigma

    def round(self, pulled_arm):
        """Simulate the current round of bidding with the given pulled arm. Returns the realization of a random normal with set mean and std."""
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])

import numpy as np


class PricingEnvironment:
    """Pricing Environment class."""
    
    def __init__(self, n_arms, prices, p, subcampaign = 1):
        """Initialize the Pricing Environment class with a number of arms, a list of prices, 
        a list of conversion rate curves for each subcampaign and the current subcampaign."""
        
        # Assignments and Initializations
        self.n_arms = n_arms
        self.probabilities = p[subcampaign]
        self.prices = prices

    def round(self, pulled_arm):
        """Simulates the reward as a Bernoulli considering the current probabilities for the pulled arm."""
        
        # print(self.probabilities)
        # print(self.prices)
        # print(self.prices[pulled_arm])
        # print(self.probabilities(self.prices[pulled_arm]), pulled_arm)
        
        # The reward is Bernoulli with probability based on the conversion rate curve for the current pulled arm
        reward = np.random.binomial(1, self.probabilities(self.prices[pulled_arm]))
        return reward

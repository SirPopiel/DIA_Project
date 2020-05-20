import numpy as np

# Pricing environment
class PricingEnvironment:
    def __init__(self, n_clicks, n_arms, list_prices, time_horizon, conversion_rate_curve):
        self.n_arms = n_arms
        # Probability of buying at the given price not caring about the subcampaign
        self.probabilities = lambda x: sum([crc(x) * nc for crc, nc in zip(conversion_rate_curve, n_clicks)])/sum(n_clicks)
        self.list_prices = list_prices
        self.time_horizon = time_horizon

    # Simulate for this round the purchases at the given pulled prices
    def round(self, pulled_arm):
        # Bernoulli with prob. second argument
        reward = np.random.binomial(1, self.probabilities(self.list_prices[pulled_arm]))
        return reward

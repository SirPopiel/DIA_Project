from learner import *
import numpy as np


class TS_Learner_Normal(Learner):
    def __init__(self, n_arms, arms, sigma, expected_mean=50):
        super().__init__(n_arms)
        self.n_arms = n_arms
        self.arms = arms
        self.sigma = sigma
        self.gaussian_parameters = np.tile([expected_mean, sigma**2], (n_arms, 1))
        self.gaussian_parameters[0, 0] = 0

    def pull_arm(self, budget):
        sampled_values = self.gaussian_parameters[:, 0] + \
                         np.random.randn(self.n_arms) * self.gaussian_parameters[:, 1]
        feasible_idxs = np.argwhere(self.arms <= budget)  # Indices which satisfy the budget allocation
        return np.argmax(sampled_values[feasible_idxs])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.gaussian_parameters[pulled_arm, 1] = 1 / (1/self.gaussian_parameters[pulled_arm, 1] + 1/self.sigma**2)
        self.gaussian_parameters[pulled_arm, 0] = self.gaussian_parameters[pulled_arm, 1] * \
                                                  ((self.gaussian_parameters[pulled_arm, 0] /
                                                    self.gaussian_parameters[pulled_arm, 1]) + reward/self.sigma**2)

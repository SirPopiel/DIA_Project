import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from learner import *

class SlidingWindowsGPTS_Learner(Learner):
    def __init__(self, n_arms, arms, window_size=30, kernel=None):
        super(SlidingWindowsGPTS_Learner, self).__init__(n_arms)
        self.arms = arms
        self.window_size = window_size
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10
        self.pulled_arms = []
        alpha = 10.0
        if not kernel:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
            n_restarts = 9
        else:
            n_restarts = 0
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                           n_restarts_optimizer=n_restarts)

    def update_observations(self, pulled_arm, reward):
        super(SlidingWindowsGPTS_Learner, self).update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def trim_window(self, what):
        if len(what) <= self.window_size:
            return what 
        return what[-self.window_size::]

    def update_model(self):
        x = np.atleast_2d(self.trim_window(self.pulled_arms)).T
        y = self.trim_window(self.collected_rewards)
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, budget):
        sampled_values = np.random.normal(self.means, self.sigmas)
        feasible_idxs = np.argwhere(self.arms <= budget)             # Indices which satisfy the budget allocation
        return np.argmax(sampled_values[feasible_idxs])

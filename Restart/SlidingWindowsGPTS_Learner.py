import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from learner import *

class SlidingWindowsGPTS_Learner(Learner):
    """Sliding Window Gaussian Process Thompson Sampling Learner inheriting from the Learner class."""
        
    def __init__(self, n_arms, arms, window_size = 30, kernel = None):
        """Initialize the Sliding-Windows Gaussian Process Thompson Sampling Learner (supercharges __init__ in Learner)."""
        
        super(SlidingWindowsGPTS_Learner, self).__init__(n_arms) # supercharge init from the learner
        
        # Assignments and Initializations
        self.arms = arms
        self.window_size = window_size
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * 10 # sigma at start is set to 10
        self.pulled_arms = []
        alpha = 10.0
        
        # When no kernel is set, Radial-basis function one is chosen with 9 restarts, otherwise no restart is needed
        if not kernel:
            # The kernel is set as the product of a constant and a Radial-basis with values 1 and range 1e-3 to 1e3
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) 
            n_restarts = 9
        else:
            n_restarts = 0
        
        # Sets the Gaussian Process Regressor from the given kernel
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=n_restarts)

    def update_observations(self, pulled_arm, reward):
        """Updates the information on the rewards keeping track of the pulled arm (supercharges update_observations in Learner)."""
        
        super(SlidingWindowsGPTS_Learner, self).update_observations(pulled_arm, reward) # supercharge update_observations from the learner
        
        # Keeps track of the pulled arm
        self.pulled_arms.append(self.arms[pulled_arm])

    def trim_window(self, what):
        """Trims the given list keeping at most the last n elements of the list, with n equal to the window size."""
        
        # No trim needed
        if len(what) <= self.window_size:
            return what 
        
        # Trim needed
        return what[-self.window_size::]

    def update_model(self):
        """Updates the model with the new means and sigmas."""
        
        # Sets the trimmed pulled arms vs rewards
        x = np.atleast_2d(self.trim_window(self.pulled_arms)).T
        y = self.trim_window(self.collected_rewards)
        
        # Fits the Gaussian process
        self.gp.fit(x, y)
        
        # Evaluates current means and sigmas with a lower bound on the standard deviation of 0.01 (for convergence)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        """Proceeds of 1 time step updating both the observations and the model."""
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self, budget):
        """Pulls the arm from the current multidimensional random normal distribution, returning the index of the best arm satisfying the budget allocation."""
        
        sampled_values = np.random.normal(self.means, self.sigmas) # pulls some random arms basing on current means and sigmas
        feasible_idxs = np.argwhere(self.arms <= budget) # finds the indices which satisfy the budget allocation
        return np.argmax(sampled_values[feasible_idxs]) # returns the index of the best arm satisfying the budget allocation

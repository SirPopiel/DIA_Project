import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from learner import *

# Gaussian Process Thompson Sampling
class GPTS_Learner(Learner):
    
    # Inits everything and calls the sklearn Gauss. Proc. Regressor
    def __init__(self, n_arms, arms):
        super(GPTS_Learner, self).__init__(n_arms) # supercharge
        self.arms = arms
        self.means = np.zeros(self.n_arms) # means at zero
        self.sigmas = np.ones(self.n_arms) * 10 # sigmas at one
        self.pulled_arms = []
        # Value added to the diagonal of the kernel matrix during fitting
        alpha = 10.0 
        # Sets constant kernel and bounds (with Radial-basis function kernel)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        # Calls the Regressor with defined params, normalizing y and restarting on 9 values the kernel interval 
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True, n_restarts_optimizer=9)

    # Adds the pulled arm to the list
    def update_observations(self, pulled_arm, reward):
        super(GPTS_Learner, self).update_observations(pulled_arm, reward) # supercharge
        self.pulled_arms.append(self.arms[pulled_arm])

    # Updates the gp on rewards(pulled_arms)
    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T # pulled armed transposed
        y = self.collected_rewards
        # Fits GP regression model predicting on all the arms
        self.gp.fit(x, y) 
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2) # min sigma is 0.01

    # Calls the updates, adding the pulled arm and using GP to predict new means and sigmas
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    # Samples on the current means/std and finds the index of the best one
    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)

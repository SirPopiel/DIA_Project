from learner import *

# Thompson Sampling Learner
class TS_Learner(Learner):
    
    # Inits the beta parameters
    def __init__(self, n_arms) :
        super().__init__(n_arms) # supercharge
        self.beta_parameters = np.ones([n_arms, 2])

    # Pulls the best arm from the random sampling
    def pull_arm(self) :
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        return idx

    # Updates the observations and parameters
    def update(self, pulled_arm, reward, price) :
        self.t += 1
        self.update_observations(pulled_arm, reward*price)
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

import numpy as np

class Learner():
    '''Basic Learner class.'''

    def __init__(self, n_arms):
        '''Initialize the Learner with a number of arms.'''

<<<<<<< HEAD
        #Assignments and Initializations
=======
        # Assignments and Initializations
>>>>>>> 95afbbd49720da0a8ebdb145dfdc0e1a2ac6f7f3
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = x = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])


    def update_observations(self, pulled_arm, reward):
        '''Updates the information on the reward of the current pulled arm.'''

        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)

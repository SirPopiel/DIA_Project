from Learner import *

class TS_Learner(Learner):
    def __init__(self, context, n_arms,  n_users) :
        super().__init__(context, n_arms)
        self.beta_parameters = np.ones([n_arms, 2])
        self.n_users = n_users

    def pull_arm(self) :
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        return idx

    def update(self, pulled_arm, rewards, price) :
        self.t += 1
        #rewards = [price*reward for reward in rewards]
        self.update_observations(pulled_arm, [price*reward for reward in rewards])
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + sum(rewards)
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + self.n_users - sum(rewards)

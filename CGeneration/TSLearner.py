from Learner import *

class TS_Learner(Learner):
    def __init__(self, context, n_arms, time) :
        super().__init__(context, n_arms, time)
        self.beta_parameters = np.ones([n_arms, 2])

    def pull_arm(self) :
        idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
        self.update_pulled_arm(idx)
        return idx

    def update(self, pulled_arm, reward, price, sc) :
        self.update_observations(pulled_arm, price*reward, sc)
        self.beta_parameters[pulled_arm, 0] += reward
        self.beta_parameters[pulled_arm, 1] += 1 - reward

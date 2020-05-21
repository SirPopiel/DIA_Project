from pricing_environment import *
from ts_learner import *
import matplotlib.pyplot as plt

class pricing_single_learner():
    
    # Init: loads pricing environment
    def __init__(self, env):
        super().__init__()
        self.env = env
        
    # Calls the TS learner updating each time the env
    def learn(self):
        self.ts_learner = TS_Learner(n_arms=self.env.n_arms) # thompson my old friend

        # One step at a time, pulls arms and updates env + learner
        for t in range(0, self.env.time_horizon):
            pulled_arm = self.ts_learner.pull_arm()
            reward = self.env.round(pulled_arm)
            
            # The reward is defined as the price of purchase (0 if no purchase)
            self.ts_learner.update(pulled_arm, reward, self.env.list_prices[pulled_arm])

        return self.ts_learner.collected_rewards
    
    # Finds the max reward price
    def find_optimal_price(self, alsolearn = True, graphics = False):
        
        # Calls the learning stuff if needed
        if alsolearn:
            self.learn()
        
        # Optimal price and reward
        self.opt_price = self.env.list_prices[np.argmax(self.ts_learner.collected_rewards)]
        self.opt_reward = max(self.ts_learner.collected_rewards)
        
        if graphics:
            self.plot_everything()
        
        return [self.opt_price, self.opt_reward]

    def plot_everything(self):
        
        plt.figure(0)
        plt.ylabel("Collected rewards during ts_learning pricing")
        plt.xlabel("t")
        plt.plot(self.ts_learner.collected_rewards, 'r.')
        
        tot_rew = []
        for price in self.env.list_prices:
            tot_rew.append(price*self.env.probabilities(price))
        plt.figure(1)
        plt.plot(self.env.list_prices, tot_rew)
        plt.xlabel("Price")
        plt.ylabel("Expected reward")
        plt.title("Expected reward given a price")
        
        plt.figure(2)
        plt.ylabel("Regret")
        plt.xlabel("t")
        plt.plot(np.cumsum(self.opt_reward - self.ts_learner.collected_rewards), 'r')
        plt.title("Cumulative regret learning the best price")
        
        plt.show()
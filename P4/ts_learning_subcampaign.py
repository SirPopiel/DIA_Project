from pricing_environment import *
from ts_learner import *


def ts_learning_subcampaign(n_arms, prices, subcampaign, time_horizon, p):
    env = PricingEnvironment(n_arms=n_arms, prices=prices, p=p, subcampaign=subcampaign)

    ts_learner = TS_Learner(n_arms=n_arms)

    for t in range(0, time_horizon):
        pulled_arm = ts_learner.pull_arm()
        reward = env.round(pulled_arm)
        ts_learner.update(pulled_arm, reward, prices[pulled_arm])

    return ts_learner.collected_rewards


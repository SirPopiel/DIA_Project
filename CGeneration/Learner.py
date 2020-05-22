from __future__ import division
import numpy as np
from ContextGenerator import hoeffding_bound

class Learner() :
    def __init__(self, context, n_arms) :
        self.context = context
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, rewards) :
        self.rewards_per_arm[pulled_arm].append(rewards)
        self.collected_rewards = np.append(self.collected_rewards, rewards)

    def context_p(self):
        cum_p = []
        for c in range(len(self.context)):
            itn = range(c,len(self.collected_rewards),len(self.context))
            cum_p.append(sum([self.collected_rewards[tosum] for tosum in itn]))
        ret = [hoeffding_bound(i/sum(self.collected_rewards),self.t) for i in cum_p]
        #ret = [i/sum(self.collected_rewards) for i in cum_p]
        #print(ret)
        return ret

    def opt_arm_reward(self):
        best_arm_full = 0
        best_arm_rew_full = 0
        best_arms = [0 for _ in self.context]
        best_arms_rew = [0 for _ in self.context]
        counter = 0
        for rwa in self.rewards_per_arm:
            if rwa:
                rwa = np.array(rwa)
                n = len(rwa)
                avg = [i/n for i in list(np.sum(rwa,0))]
                ## whole context
                avg_full = hoeffding_bound(sum(avg),n)
                if avg_full > best_arm_rew_full:
                    best_arm_full = counter
                    best_arm_rew_full = avg_full
                ## Separating subcampaigns
                avg_lb = [hoeffding_bound(i,n) for i in avg]
                subcounter = 0
                for x in avg_lb:
                    if x > best_arms_rew[subcounter]:
                        best_arms[subcounter] = counter
                        best_arms_rew[subcounter] = x
                    subcounter += 1
            counter += 1

        return best_arms, best_arms_rew, best_arm_full, best_arm_rew_full/len(self.context)

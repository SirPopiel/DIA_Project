import numpy as np
import math

def hoeffding_bound(x, n, confidence = 0.90):
    return max(x - math.sqrt((-math.log(confidence))/n), 0)

def sigma_bound(x, n, sigma):
    return max(x - math.sqrt(2*sigma/n), 0)


class Learner() :
    def __init__(self, context, n_arms, time) :
        self.context = context
        self.n_arms = n_arms
        self.t = time
        self.rewards_per_arm = self.init_rewards(n_arms)
        self.collected_rewards = dict()
        self.pulled_arms = {arm : 0 for arm in range(n_arms)}
        self.users_per_arm = self.init_rewards(n_arms)

    def update_observations(self, pulled_arm, reward, sc) :
        self.rewards_per_arm[pulled_arm][sc].append(reward)
        self.collected_rewards[self.t][sc] += reward

    def init_rewards(self,n_arms):
        ret = dict()
        for i in range(self.n_arms):
            ret.update({i : {sc : [] for sc in self.context}})
        return ret

    def update_status(self):
        self.t = self.t + 1
        self.collected_rewards.update({self.t : {sc : 0 for sc in self.context}})

    def update_pulled_arm(self, pulled_arm):
        self.pulled_arms[pulled_arm] += 1

    def best_performance(self):
        best_rewards = {sc : (0, np.nan, 0) for sc in self.context}
        if len(self.context) == 3:
            couples = ['12', '13', '23']
            best_reward_total = {k : (0, np.nan, 0) for k in couples}
        if len(self.context) == 2:
            couple = str(self.context[0])+str(self.context[1])
            best_reward_total = {couple : (0, np.nan, 0)}
        for arm in range(self.n_arms):
            n = self.pulled_arms[arm]
            if len(self.context) == 3:
                for couple in [[1,2], [1,3], [2,3]]:
                    strcouple = str(couple[0])+str(couple[1])
                    if n:
                        couple_rewards = [sum(self.rewards_per_arm[arm][sc]) for sc in couple]
                        sm = sum(couple_rewards)
                        sigma = np.std(couple_rewards)
                        avg_total = sigma_bound(sm/n, n, sigma) #sm/n
                        if avg_total > best_reward_total[strcouple][0]:
                            best_reward_total[strcouple] = (avg_total, arm, n)
            if len(self.context) == 2:
                couple = str(self.context[0])+str(self.context[1])
                if n:
                    temp_reward = [sum(i) for i in list(self.rewards_per_arm[arm].values())]
                    sm = sum(temp_reward)
                    sigma = np.std(temp_reward)
                    avg_total = sigma_bound(sm/n, n, sigma)
                    if avg_total > best_reward_total[couple][0]:
                        best_reward_total[couple] = (avg_total, arm, n)

            for sc in self.context:
                if n:
                    sm = sum(self.rewards_per_arm[arm][sc])
                    sigma = np.std(self.rewards_per_arm[arm][sc])
                    avg_rew = sigma_bound(sm/n, n, sigma)
                    if avg_rew > best_rewards[sc][0]:
                        best_rewards[sc] = (avg_rew, arm, n)
            #print(best_rewards, best_reward_total)
        return best_rewards, best_reward_total

    def update_users_per_arm(self, pulled_arm, sc, buyers):
        self.users_per_arm[pulled_arm][sc].append(buyers)

    def compute_dynamic_probabilities(self, best_rewards, clicks):
        #print(best_rewards, clicks, self.users_per_arm)
        return {sc : sum(self.users_per_arm[best_rewards[sc][1]])/clicks[sc] for sc in self.context}

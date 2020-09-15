from PricingEnvironment import *
from TSLearner import *
import numpy as np
import collections

def is_split(context, best_rewards, best_reward_total, context_probabilities):
    def split_condition(left_rew, right_rew, probabilities):
        return left_rew[0]+left_rew[1]-right_rew    #*probabilities[0]

    if len(context) == 3:
        couples = ['12', '13', '23']
        split_values = {couple : 0 for couple in couples}
        for couple in couples:
            sc = [int(couple[0]), int(couple[1])]
            temp_prob = list(context_probabilities.values())
            norm_probabilities = [prob/sum(temp_prob) for prob in temp_prob]
            split_values[couple] = split_condition([best_rewards[s][0] for s in sc], best_reward_total[couple][0], norm_probabilities)

        sorted_split_values = collections.OrderedDict(sorted(split_values.items(), key=lambda t: t[1]))
        #print(sorted_split_values)
        best_arms = {s : best_rewards[s][1] for s in context}
        if best_arms[1] == best_arms[2] and best_arms[2] == best_arms[3]:
            return []
        idx = -1
        to_divide = list(sorted_split_values.keys())[idx]
        if best_arms[int(to_divide[0])] == best_arms[int(to_divide[1])]:
            idx -= 1
            to_divide = list(sorted_split_values.keys())[idx]
        to_mantain = list(sorted_split_values.keys())[0]
        M = list(sorted_split_values.values())[idx]
        if M < 0:
            return []
        else:
            for sc in to_divide:
                if sc not in to_mantain:
                    c1 = [int(sc)]
            return { 'B' : c1, 'C' : [int(to_mantain[0]), int(to_mantain[1])]}

    if len(context) == 2:
        couple = str(context[0])+str(context[1])
        best_arms = [best_rewards[s][1] for s in context]
        if best_arms[0] != best_arms[1]:
            temp_prob = list(context_probabilities.values())
            norm_probabilities = [prob/sum(temp_prob) for prob in temp_prob]
            split_value = split_condition([best_rewards[s][0] for s in context], best_reward_total[couple][0], norm_probabilities)
            if split_value < 0:
                return []
            else:
                return {'D' : [context[0]], 'E' : [context[1]]}
        else:
            return []

def update_contexts(new_contexts, contexts, learners, environments, n_arms, prices, context_probabilities, conversion_rate, time):
    if not new_contexts:
        return contexts, learners, environments
    else:
        contexts.update(new_contexts)
        if len(contexts) == 3:
            del contexts['A']
            del environments['A']
            del learners['A']
            learners.update({'B' : TS_Learner(new_contexts['B'], n_arms, time)})
            environments.update({'B' : PricingEnvironment(new_contexts['B'], prices, conversion_rate)})
            learners.update({'C' : TS_Learner(new_contexts['C'], n_arms, time)})
            environments.update({'C' : PricingEnvironment(new_contexts['C'], prices, conversion_rate)})
        elif len(contexts) == 4:
            del contexts['C']
            del environments['C']
            del learners['C']
            learners.update({'D' : TS_Learner(new_contexts['D'], n_arms, time)})
            environments.update({'D' : PricingEnvironment(new_contexts['D'], prices, conversion_rate)})
            learners.update({'E' : TS_Learner(new_contexts['E'], n_arms, time)})
            environments.update({'E' : PricingEnvironment(new_contexts['E'], prices, conversion_rate)})
        print(contexts)
        return contexts, learners, environments

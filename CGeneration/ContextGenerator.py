import itertools
import math
'''
def subsets(S):
    subs = []
    for m in range(1,len(S)):
        subs += list(map(set, itertools.combinations(S,m)))
    return subs
'''
### Compute lower bound with hoeffding formula
def hoeffding_bound(x, n, confidence = 0.95):
    return max(x - math.sqrt((-math.log(confidence))/n),0)
### chech if split and in case split
def split(context, best_arms_rew, best_arm_rew_full, context_probabilities):
    if len(context) == 3:
        values = [best_arms_rew[c]*context_probabilities[c] for c in range(3)]
        if sum(values) > best_arm_rew_full:
            ret = values.index(max(values))
            context.pop(ret)
            #print([[ret], context])
            return True, [[ret], context]

    else:
        #print(sum([best_arms_rew[c]*context_probabilities[c] for c in range(2)]), best_arm_rew_full)
        if sum([best_arms_rew[c]*context_probabilities[c] for c in range(2)]) > best_arm_rew_full:
            return True, [[c] for c in context]

    return False, []

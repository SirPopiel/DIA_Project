import pulp


def get_reward(i, sub, rewards_per_subcampaign):
    '''Returns the reward at the given index for the given subcampaign from the list of list of rewards.'''
    
    return rewards_per_subcampaign[sub - 1][i]


def get_budget(i, budgets): # ???
    '''Returns the budget at the current index'''
    
    return budgets[i]


def good_knapsack(list_budgets, rewards_per_subcampaign, budget):
    '''Finds the optimal allocation as a solution of a knapsack problem. It considers a list of budgets, rewards per subcampaing, current budget.'''
    
    n_arms = len(list_budgets) # finds the number of arms

    # Defines the choice for the LP problem
    sub_1_choice = pulp.LpVariable.dicts('sub_1_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    sub_2_choice = pulp.LpVariable.dicts('sub_2_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    sub_3_choice = pulp.LpVariable.dicts('sub_3_choice', [i for i in range(n_arms)],
                                         lowBound=0,
                                         upBound=1,
                                         cat=pulp.LpInteger)

    # The LP problem appears
    p1_model = pulp.LpProblem("P1_Model", pulp.LpMaximize)
    p1_model += (
            sum([get_reward(choice, 1, rewards_per_subcampaign) * sub_1_choice[choice] for choice in range(n_arms)]) +
            sum([get_reward(choice, 2, rewards_per_subcampaign) * sub_2_choice[choice] for choice in range(n_arms)]) +
            sum([get_reward(choice, 3, rewards_per_subcampaign) * sub_3_choice[choice] for choice in range(n_arms)])
    )

    # Constraints, budget on one <= total budget = 1
    p1_model += sum([sub_1_choice[choice] for choice in range(n_arms)]) <= budget
    p1_model += sum([sub_2_choice[choice] for choice in range(n_arms)]) <= budget
    p1_model += sum([sub_3_choice[choice] for choice in range(n_arms)]) <= budget

    # Constraints, total budget <= budget
    p1_model += (
                        sum([get_budget(choice, list_budgets) * sub_1_choice[choice] for choice in range(n_arms)]) +
                        sum([get_budget(choice, list_budgets) * sub_2_choice[choice] for choice in range(n_arms)]) +
                        sum([get_budget(choice, list_budgets) * sub_3_choice[choice] for choice in range(n_arms)])
                ) <= budget

    # Here the magic happens
    p1_model.solve()

    new_allocations = {}

    # For each arm gets the reward of the chosen subcampaign
    for choice in range(n_arms):
        if sub_1_choice[choice].value() == 1.0:
            new_allocations[1] = choice #list_budgets[choice]  # 
        if sub_2_choice[choice].value() == 1.0:
            new_allocations[2] = choice# list_budgets[choice]  # choice
        if sub_3_choice[choice].value() == 1.0:
            new_allocations[3] = choice #list_budgets[choice]  # choice

    # Returns the allocation
    return new_allocations

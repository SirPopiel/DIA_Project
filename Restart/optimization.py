import numpy as np


# return mode: 0 _ only print; 1 : clicks for budget_index + distribution per sub, 2 : table and table of subs
def dynamic_opt(budget_list, budget_index, rewards_per_subcampaign, return_mode=1, verbose=False):
    nrows = len(rewards_per_subcampaign[:]) + 1
    ncols = len(budget_list)

    table = np.zeros([nrows, ncols])  # here i store the dynamic programming intermediate results
    table_sub = np.zeros([nrows, ncols, nrows - 1])  # here the split with which we get the results

    # NB: if we want to optimize memory consumption, we actually only need two rows forever for table and table sub

    # Dynamic programming algorithm
    temp = 0
    for i in range(1, nrows):
        for j in range(ncols):
            temp_vec = []  # temporary vector with partial results for every combination
            temp_sub = []  # same with subcampaigns split
            for k in range(j + 1):
                subs = np.zeros(nrows - 1)
                temp = rewards_per_subcampaign[i - 1][j - k] + table[i - 1][
                    k]  # inversely cycle through vecs to get partial results
                temp_vec.append(temp)
                for sub in range(i - 1):  # I have to keep track of the split for subcampaigns
                    subs[sub] = table_sub[i - 1][k][sub]
                subs[i - 1] = budget_list[j - k]
                temp_sub.append(subs)

            table[i, j] = max(temp_vec)  # get the max and put it in the table
            index = np.argmax(temp_vec)  # get the index and put the corresponding split in the table of splits

            for sub in range(nrows - 1):  # numero subcampagne
                table_sub[i, j, sub] = temp_sub[index][sub]

    if return_mode == 0:
        print(table)
        print(table_sub)

    if return_mode == 1:
        spesa = []
        for i in range(nrows - 1):
            spesa.append(table_sub[nrows - 1][budget_index][i])
        if verbose:
            print(spesa)
        return spesa
        # return(table[nrows-1][budget_index] , spesa)

    if return_mode == 2:
        return table, table_sub

    return


# stupid test main

def main():
    rewards_per_subcampaign = [[0, 2, 3, 8, 8], [0, 4, 4, 4, 4], [0, 3, 5, 6, 7]]
    budget_list = [0, 10, 20, 30, 40]
    budget_index = 4

    dynamic_opt(budget_list, budget_index, rewards_per_subcampaign, 1)


if __name__ == "__main__":
    main()

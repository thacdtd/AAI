from random import *


def crp(num_customers, alpha):
    table_assignments = []
    if num_customers <= 0:
        return table_assignments
    table_assignments = [1]
    next_open_table = 2

    for i in range(1, num_customers):
        rand = uniform(0, 1)
        if rand < alpha*1.0 / (alpha + i - 1):
            table_assignments.append(next_open_table)
            next_open_table += 1
        else:
            which_table = table_assignments[randint(0, len(table_assignments) - 1)]
            table_assignments.append(which_table)

    return table_assignments

if __name__ == "__main__":
    print crp(num_customers=10, alpha=2)

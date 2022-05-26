import numpy as np
from scipy.stats import multinomial, norm
from itertools import product
import pytest

def simulate_data(possible_locations, variance_conditions):

    """
        Args:
            possible_locations --> list; of poss locs
            variance_conditions --> list; off sds to paramterize audition
    """

    def bin(arr, possible_locations = possible_locations):
        return [min(possible_locations, key=lambda x:abs(x-i)) for i in arr]

    Vlist, Alist = possible_locations, possible_locations

    combinations = list(product(Vlist, Alist, variance_conditions))
    dataset = {i:None for i in combinations}

    for cond in dataset.keys():
        v, a = norm.rvs(loc = cond[0], scale = 5, size = 10000), norm.rvs(loc = cond[1], scale = cond[2], size = 10000)
        vBin, aBin = bin(v), bin(a)
        vCounts, aCounts = [vBin.count(i) for i in possible_locations], [aBin.count(i) for i in possible_locations]
        dataset[cond] = [vCounts, aCounts]

    return dataset


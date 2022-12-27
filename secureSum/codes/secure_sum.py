# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:21:40 2022

@author: utkuk
"""
import numpy as np

def random_split(s_vector, parties, shared_resources):
    a = [np.random.random(parties) for i in range(shared_resources)]
    a = [a[i]/a[i].sum() for i in range(len(a))]
    a = [a[i]*s_vector[i] for i in range(len(a))]
    return a


def secure_sum(parties, shared_resources, s_values, iteration):
    for _ in range(iteration):
        random_parties = [random_split(s_values[i], parties, shared_resources) 
                          for i in range(parties)]
        s_sum = np.zeros(shared_resources)
        for i in range(shared_resources):
            for j in range(parties):
                for k in range(parties):
                    s_sum[i] += random_parties[j][i][k]
    return s_sum


def main():
    import numpy as np
    from timeit import default_timer as timer
    data = []
    replications = 50
    
    for parties in [5, 8, 10, 20, 50, 100]:
        for shared in [5, 10, 20, 50, 100]:
            for iteration in [50, 100, 200, 500, 1000]:
                for _ in range(replications):
                    s_values = np.random.rand(parties, shared)
                    start = timer()
                    _ = secure_sum(parties, shared, s_values, iteration)
                    end = timer()
                    time = end - start
                    data.append([parties, shared, iteration, time])
                print('Parties: {} Shared: {} Iteration: {}'.format(parties, 
                                                                    shared,
                                                                    iteration))
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv('results_smc.csv')

if __name__ == "__main__":
    main()
    

    
    
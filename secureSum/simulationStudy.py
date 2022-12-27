# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:19:56 2022

@author: utkuk
"""

def init_model(ins):
    import numpy as np
    import codes.model_original as model_original
    import codes.model_sk as model_sk             # Model with s_k introduced
# Coefficient for demand vectors
    demand_coef = 0.05          # each products' demand (d_k) will be multiplied with this
# Upper Limit creation for each party
    # if np.size(upper_limit_coef) > 1:
    #     upper_limit = upper_limit_coef
    # elif np.size(upper_limit_coef) == 1:
    #     upper_limit = upper_limit_coef * np.ones(parties)
# Original model
    original_model = model_original.optimize(ins, np.zeros(sum(ins.n)))
    # get optimal values of x's
    optimal_x = np.zeros(sum(ins.n))
    for i in range(sum(ins.n)):
        optimal_x[i] = original_model.getVars()[i].x

# Create demand vector
    product_demand = np.zeros(sum(ins.n))
    for i in range(sum(ins.n)):
        if optimal_x[i] > 0:
            product_demand[i] = optimal_x[i] * (1 - demand_coef * np.random.random())

    product_demand = product_demand * -1
# Solve the original model with demand added
    original_model_demand = model_original.optimize(ins, product_demand)


# Initial model where everybody input their information to the model, also s_k is introduced
    init_mod = model_sk.optimize(ins, product_demand)

    return original_model, original_model_demand, init_mod, product_demand


def dataPrivate_model(ins, step_len, init_mod, beta):
    import numpy as np
    import codes.model_distr as model_distr
# Coefficients
    max_iter_subgradient = 400
# Upper Limit creation for each party
    # if np.size(upper_limit_coef) > 1:
    #     upper_limit = upper_limit_coef
    # elif np.size(upper_limit_coef) == 1:
    #     upper_limit = upper_limit_coef * np.ones(parties)
# Subgradient algorithm
    # lambda is initialized as 0 vector
    # while not coverged
    #   each party solves its subproblem, returns s_k
    #   if better solution obtained, then it is stored
    #   lambda values are updated
    lamda = np.median(ins.r)*np.zeros(ins.m)
    lamda_vals = []
    lamda_vals.append(lamda)
    iteration = 1
    best_obj_vals = []
    best_obj_val = init_mod.objVal * 1000
    
    optimal_lamdas = init_mod.Pi[-ins.m:]
    M_sq = np.linalg.norm(optimal_lamdas)**2
    c_norm_sq = np.linalg.norm(ins.c)**2
    B_sq = (ins.parties - 1)**2*c_norm_sq
    
    while iteration <= max_iter_subgradient:
        # Solves subproblems and returns the summation of s_k's and each submodels
        (opt_mod, s_total, x_values,
         private_cap_duals,
         common_cap_duals) = model_distr.optDist(ins, lamda)

        # Keeps the best solution until now
        if sum(opt_mod[i].objVal for i in range(ins.parties)) + np.inner(ins.c, lamda) < best_obj_val:
            best_obj_val = sum(opt_mod[i].objVal for i in range(ins.parties)) + np.inner(ins.c, lamda)
            opt_mod_min = opt_mod
            min_iter = iteration
            # best_s_total = s_total
            # best_lamda = lamda
            best_x_val = x_values
            best_private_cap_duals = private_cap_duals
            best_common_cap_duals = common_cap_duals
        best_obj_vals.append(best_obj_val)
        
        # Lambda updates
        gradient = s_total - ins.c
        if max(abs(gradient)) <= 1.0e-6:
            break

        if iteration == 1:
            scaler = 1
        
        if step_len == 1:
            nu = 1.0/(iteration**(0.5))
        elif step_len == 2:
            nu = scaler * np.sqrt(M_sq)/(np.sqrt(B_sq*max_iter_subgradient))

        if iteration > 2:
            lamda = np.maximum(0, lamda + nu * gradient + beta * (lamda - lamda_vals[-2]))
        else:
            lamda = np.maximum(0, lamda + nu * gradient)
        ###
        iteration = iteration + 1
        lamda_vals.append(lamda)

    return best_obj_vals, best_obj_val, best_x_val, best_private_cap_duals, best_common_cap_duals, lamda_vals



def simulation(parameter_list):
    import numpy as np
    from codes.tools import instance
    parties = int(parameter_list[0])
    m = int(parameter_list[1])
    momentumParam = parameter_list[2]
    seed = int(parameter_list[3])
    step_len = int(parameter_list[4])
    max_private_cap = 10
    max_product = 15

    # Create data
    ins = instance(parties, m, max_private_cap, max_product, seed)
    # n, m_parties, r, A, B, c, c_individual = data_creation(parties, m,
    #                                                        max_private_cap,
    #                                                        max_product,
    #                                                        seed)
    # Solve initial model with and without demand
    original_model, original_model_demand, init_mod, ins.product_demand = init_model(ins)
    optimal_lamdas = init_mod.Pi[-ins.m:]
    # Solve data private model
    (obj_vals,
     best_obj_val,
     best_x_val,
     best_private_cap_duals,
     best_common_cap_duals,
     lamda_vals) = dataPrivate_model(ins, step_len,
                                     init_mod,
                                     beta=momentumParam)
    obj_vals = [(x-original_model.objVal)/original_model.objVal for x in obj_vals]
    obj_vals.insert(0, seed)
    obj_vals.insert(1, step_len)
    obj_vals.insert(2, parties)
    obj_vals.insert(3, momentumParam)
    print('Parties: {} Seed: {} MomentumParam: {}'.format(parameter_list[0], 
                                                          parameter_list[3],
                                                          parameter_list[2]))
    if 0 in optimal_lamdas:
        optimal_lamdas = [i + 0.01 for i in optimal_lamdas]
    lamda_comp = [np.linalg.norm((optimal_lamdas-i)) for i in lamda_vals]
    return obj_vals, lamda_comp
    # with open('./results/momentum_dataPrivate.csv', 'a+', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(obj_vals)

if __name__ == "__main__":
    from codes.tools import cartesian
    grid = list(cartesian(([5, 8, 10, 20],              # number of parties
                           [5],                         # number of shared capacities
                           [0, 0.25, 0.50, 0.75, 0.90], # momentum parameter
                           range(1, 51),
                           [1, 2])))  # seed

    # for i in range(len(grid)):
    #     simulation(grid[i])
    #     print('Parties: {} Seed: {}'.format(grid[i][0], grid[i][3]))

    from dask.distributed import Client
    client = Client(n_workers=4)
    results = []
    for i in range(len(grid)):
        future = client.submit(simulation, grid[i], pure=False)
        results.append(future)
    
    results_last = client.gather(results)
    objs = [i[0] for i in results_last]
    lamdas = [i[1] for i in results_last]
    client.close()
    import pandas as pd
    df = pd.DataFrame(objs)
    df.to_csv('results_obj_comparison_last.csv')
    df = pd.DataFrame(lamdas)
    df.to_csv('results_lamdas_comparison_last.csv')
    

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:57:29 2024

@author: 20224695
"""

import numpy as np
from scipy.stats import poisson
import pandas as pd
from scipy import linalg
import time 


num_actions = 2

K_range = [240, 260, 280, 300]
I_range = [26, 30, 34, 38]
L_range = [4,5,6,7]
p_range = [15]
s_range = [4]
lambda_c_range = [10, 12, 14, 16]
P_b_range = [0, 0.1, 0.2, 0.3]


# Prepare a DataFrame to store results
results = []
case = 0
epsilon = 1e-6
gamma = 0.999

# Arrival probability functions
def arrival_probability(L_i, k):
    arr = lambda_c * (L_i / L)
    return poisson.pmf(k, arr)

def arr_greater_than(L_i, k):
    arr = lambda_c * (L_i / L)
    return 1 - poisson.cdf(k, arr)

def calculate_stationary_distribution(P, policy, I, L):
    num_states = (I + 1) * (L + 1) * (L + 1)
    P_flat = np.zeros((num_states, num_states))
    
    for i in range(I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if l >= l_b:
                    continue
                state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
                action = policy[i, l, l_b]
                
                for i_next in range(I + 1):
                    for l_next in range(L + 1):
                        for l_b_next in range(L + 1):
                            next_state_index = i_next * (L + 1) * (L + 1) + l_next * (L + 1) + l_b_next
                            P_flat[state_index, next_state_index] = P[i, l, l_b, action, i_next, l_next, l_b_next]
    
    row_sums = P_flat.sum(axis=1)
    
    zero_sum_rows = (row_sums == 0)
    P_flat[zero_sum_rows, :] = 1 / num_states
    row_sums[zero_sum_rows] = 1
    
    P_flat = P_flat / row_sums[:, np.newaxis]
    
    if np.any(np.isnan(P_flat)) or np.any(np.isinf(P_flat)):
        P_flat = np.nan_to_num(P_flat, nan=0, posinf=0, neginf=0)
        row_sums = P_flat.sum(axis=1)
        P_flat = P_flat / row_sums[:, np.newaxis]
    
    eigenvalues, eigenvectors = linalg.eig(P_flat.T)
    index = np.argmin(np.abs(eigenvalues - 1))
    stationary_dist = eigenvectors[:, index].real
    stationary_dist = stationary_dist / np.sum(stationary_dist)
    
    stationary_dist_shaped = np.zeros((I + 1, L + 1, L + 1))
    for i in range(I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if l >= l_b:
                    continue
                state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
                stationary_dist_shaped[i, l, l_b] = stationary_dist[state_index]
    
    return stationary_dist_shaped

for K in K_range:
    for I in I_range:
        for L in L_range:
            for p in p_range:
                for s in s_range:
                    for lambda_c in lambda_c_range:
                        for P_b in P_b_range:
                            case += 1
                            if case < 679:
                                continue
                            start_time = time.time()
                            # Reward array
                            R = np.zeros((I + 1, L + 1, L + 1, num_actions))
                            for i in range(I + 1):
                                for l in range(L + 1):
                                    for l_b in range(L + 1):
                                        for a in range(num_actions):
                                            if a == 1:
                                                R[i, l, l_b, a] = - K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
                                            else:
                                                R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p
                            print('Rewards completed')
                            # Transition probability matrix
                            P = np.zeros((I + 1, L + 1, L + 1, num_actions, I + 1, L + 1, L + 1))
                            for i in range(I + 1):
                                for l in range(L + 1):
                                    for l_b in range(L + 1):
                                        if l_b <= l and l != 0:
                                            continue
                                        else:
                                            if i == 0 and l == 0 and l_b == 0:
                                                P[i, l, l_b, 0, i, l, l_b] = 1
                                                P[i, l, l_b, 1, i, l, L] = 1
                                            elif i == 0 and l == 0 and l_b != 0:
                                                P[i, l, l_b, 0, i, l, l_b - 1] = P_b
                                                P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
                                                for k in range(I+1):    
                                                    P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
                                                P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                            elif i == 0 and l != 0 and l_b != 0:
                                                P[i, l, l_b, 0, i, l - 1, l_b - 1] = P_b
                                                P[i, l, l_b, 0, i, l - 1, l_b] = 1 - P_b
                                                for k in range(I+1):
                                                    P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
                                                P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                            elif i != 0 and l == 0 and l_b == 0:
                                                P[i, l, l_b, 0, i, l, l_b] = 1
                                                P[i, l, l_b, 1, 0, l_b, L] = 1
                                            elif i != 0 and l == 0 and l_b != 0:
                                                P[i, l, l_b, 0, i, l, l_b - 1] = P_b
                                                P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
                                                for k in range(I+1):
                                                    P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
                                                P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                            else:
                                                for k in range(I+1):
                                                    P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
                                                P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                                
                                                for k in range(i+1):
                                                    P[i, l, l_b, 0, i - k, l - 1, l_b - 1] = arrival_probability(l, k)*P_b
                                                P[i, l, l_b, 0, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1)*P_b
                                                
                                                for k in range(i+1):
                                                    P[i, l, l_b, 0, i - k, l - 1, l_b] = arrival_probability(l, k)*(1 - P_b)
                                                P[i, l, l_b, 0, 0, l - 1, l_b] = arr_greater_than(l, i - 1)*(1 - P_b)
                            print('Probabilities completed')       
                            
                            double_threshold_averages = np.zeros((I+1, L+1))
                            
                            for i_t in range(I+1):
                                for l_t in range(L+1):
                                    double_threshold_policy = np.zeros((I+1, L+1, L+1))
                                    double_threshold_policy[0, :, :] = 1
                                    double_threshold_policy[:, 0, :] = 1
                                    for i in range(I+1):
                                        for l in range(L+1):
                                            if i < i_t and l < l_t:
                                                double_threshold_policy[i, l, :] = 1
                                    double_threshold_averages[i_t, l_t] = stationary_dist
                                                
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                
                            })
                           
                            print(case)

# # Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("poisson_with_gamma_0.999_3.xlsx", index=False)

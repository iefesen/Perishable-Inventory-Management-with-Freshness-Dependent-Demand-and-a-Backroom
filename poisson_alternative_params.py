# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:29:35 2024

@author: 20224695
"""

import numpy as np
from scipy.stats import poisson
# from scipy.stats import nbinom
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
gamma = 0.99

# Arrival probability functions
def arrival_probability(L_i, k):
    arr = lambda_c * (L_i / L)
    return poisson.pmf(k, arr)

def arr_greater_than(L_i, k):
    arr = lambda_c * (L_i / L)
    return 1 - poisson.cdf(k, arr)

for K in K_range:
    for I in I_range:
        for L in L_range:
            for p in p_range:
                for s in s_range:
                    for lambda_c in lambda_c_range:
                        for P_b in P_b_range:
                            
                            case += 1

                            start_time_optimal = time.time()
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
                            
                            # Value iteration parameters
                            
                            V = np.zeros((I+1,L+1, L+1))
                            V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # Value iteration
                            iteration = 0
                            while True:
                                V_prev = np.copy(V)
                                iteration += 1
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                for a in range(num_actions):
                                                    if (i == 0 or l == 0) and a == 0:
                                                        V_actions[i, l, l_b, a] = -float('inf')
                                                    else:
                                                        V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                                V = np.max(V_actions, axis = 3)
                                
                                if np.max(np.abs(V - V_prev)) < epsilon:
                                    print('Value iteration completed at iteration: ', iteration)
                                    break
                                
                            optimal_policy = np.argmax(V_actions, axis = 3)
                            end_optimal = time.time()
                            optimal_time = end_optimal - start_time_optimal
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

                            # Calculate optimal average reward
                            
                            stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            optimal_avg_reward = 0
                            for i in range(I+1):
                                for l in range(L+1):
                                    for l_b in range(L+1):
                                        if l >= l_b:
                                            continue
                                        else:
                                            optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]
                            
                            end_time_optimal = time.time()
                            optimal_time = end_time_optimal - start_time_optimal
                            # Calculate threshold policy results for i
                            def calculate_avg_reward_for_threshold_i(threshold, I, L, P, R):
                                # Create threshold policy
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                            
                                # Calculate stationary distribution for the current threshold
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                            
                                # Calculate the average reward for the current threshold
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                            
                                return threshold_avg_reward
                            
                            def find_max_threshold_i_bisection(I):
                                low, high = 0, I
                                iteration_i = 0
                                while high != low:
                                    # Calculate midpoints
                                    mid1 = low + (high - low) / 3
                                    mid2 = high - (high - low) / 3
                                    iteration_i = iteration_i + 1
                                    # Calculate rewards only for these midpoints
                                    reward_mid1 = calculate_avg_reward_for_threshold_i(int(mid1), I, L, P, R)
                                    reward_mid2 = calculate_avg_reward_for_threshold_i(int(mid2), I, L, P, R)
                            
                                    print(f"Threshold {int(mid1)} Avg Reward: {reward_mid1}")
                                    print(f"Threshold {int(mid2)} Avg Reward: {reward_mid2}")
                            
                                    if reward_mid1 < reward_mid2:
                                        low = mid1  # The max is in the range (mid1, high)
                                    else:
                                        high = mid2  # The max is in the range (low, mid2)
                                    if int(mid1) == int(mid2) :
                                        break
                            
                                # Best threshold is at the midpoint of the final range
                                best_threshold_i = int((low + high) / 2)
                                best_avg_reward_i = calculate_avg_reward_for_threshold_i(best_threshold_i, I, L, P, R)
                            
                                # print(f"Best Threshold: {best_threshold} with Avg Reward: {best_avg_reward}")
                                return best_threshold_i, best_avg_reward_i, iteration_i
                            
                            # Calculate threshold policy results for l
                            def calculate_avg_reward_for_threshold_l(threshold, I, L, P, R):
                                # Create threshold policy
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:, :threshold + 1, :] = 1
                                threshold_policy[0, :, :] = 1
                            
                                # Calculate stationary distribution for the current threshold
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                            
                                # Calculate the average reward for the current threshold
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                            
                                return threshold_avg_reward
                            
                            def find_max_threshold_l_bisection(L):
                                low, high = 0, L
                                iteration_l = 0
                                while high != low:
                                    # Calculate midpoints
                                    mid1 = low + (high - low) / 3
                                    mid2 = high - (high - low) / 3
                                    iteration_l = iteration_l + 1
                                    # Calculate rewards only for these midpoints
                                    reward_mid1 = calculate_avg_reward_for_threshold_l(int(mid1), I, L, P, R)
                                    reward_mid2 = calculate_avg_reward_for_threshold_l(int(mid2), I, L, P, R)
                            
                                    print(f"Threshold {int(mid1)} Avg Reward: {reward_mid1}")
                                    print(f"Threshold {int(mid2)} Avg Reward: {reward_mid2}")
                            
                                    if reward_mid1 < reward_mid2:
                                        low = mid1  # The max is in the range (mid1, high)
                                    else:
                                        high = mid2  # The max is in the range (low, mid2)
                                    if int(mid1) == int(mid2) :
                                        break
                            
                                # Best threshold is at the midpoint of the final range
                                best_threshold_l = int((low + high) / 2)
                                best_avg_reward_l = calculate_avg_reward_for_threshold_l(best_threshold_l, I, L, P, R)
                            
                                # print(f"Best Threshold: {best_threshold} with Avg Reward: {best_avg_reward}")
                                return best_threshold_l, best_avg_reward_l, iteration_l
                            
                            best_threshold_i, best_avg_reward_i, iteration_i = find_max_threshold_i_bisection(I)
                            print('Threshold i found')
                            best_threshold_l, best_avg_reward_l, iteration_l = find_max_threshold_l_bisection(L)
                            print('Threshold l found')
                            convergence_time = time.time()
                            end_time = convergence_time - end_time_optimal
                                                            
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                'Optimal_avg': optimal_avg_reward,
                                'Optimal_policy_time': optimal_time,
                                'Best_threshold_i': best_threshold_i,
                                'Best_average_i': best_avg_reward_i,
                                'Best threshold_l': best_threshold_l,
                                'Best_average_l': best_avg_reward_l, 
                                'Convergence_time': end_time,
                                'Iteration_i': iteration_i,
                                'Iteration_l': iteration_l,
                                
                            })
                           
                            print(case)

# # Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("master_file_alternative_parameters.xlsx", index=False)

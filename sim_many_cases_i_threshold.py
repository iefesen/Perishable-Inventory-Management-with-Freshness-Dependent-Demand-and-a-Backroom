import numpy as np
from scipy.stats import poisson
import pandas as pd
from scipy import linalg


# Define parameter ranges
# K_range = np.arange(90000, 100000, 1000)
# I_range = [80]
# L_range = [7]
# p_range = [1500]
# s_range = [400]
# lambda_c_range = [20]
# P_b_range = [0.1]

# num_actions = 2
# np.random.seed(42)

# # Prepare a DataFrame to store results
# results = []
# case = 0
# epsilon = 1e-6
# gamma = 0.99

# # Arrival probability functions
# def arrival_probability(L_i, k):
#     arr = lambda_c * (L_i / L)
#     return poisson.pmf(k, arr)

# def arr_greater_than(L_i, k):
#     arr = lambda_c * (L_i / L)
#     return 1 - poisson.cdf(k, arr)

# for K in K_range:
#     for I in I_range:
#         for L in L_range:
#             for p in p_range:
#                 for s in s_range:
#                     for lambda_c in lambda_c_range:
#                         for P_b in P_b_range:
#                             case += 1
#                             # Reward array
#                             R = np.zeros((I + 1, L + 1, L + 1, num_actions))
#                             for i in range(I + 1):
#                                 for l in range(L + 1):
#                                     for l_b in range(L + 1):
#                                         for a in range(num_actions):
#                                             if a == 1:
#                                                 R[i, l, l_b, a] = - K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
#                                             else:
#                                                 R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p
#                             print('Rewards completed')
#                             # Transition probability matrix
#                             P = np.zeros((I + 1, L + 1, L + 1, num_actions, I + 1, L + 1, L + 1))
#                             for i in range(I + 1):
#                                 for l in range(L + 1):
#                                     for l_b in range(L + 1):
#                                         if l_b <= l and l != 0:
#                                             continue
#                                         else:
#                                             if i == 0 and l == 0 and l_b == 0:
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1
#                                                 P[i, l, l_b, 1, i, l, L] = 1
#                                             elif i == 0 and l == 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
#                                                 for k in range(I+1):    
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             elif i == 0 and l != 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l - 1, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l - 1, l_b] = 1 - P_b
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             elif i != 0 and l == 0 and l_b == 0:
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1
#                                                 P[i, l, l_b, 1, 0, l_b, L] = 1
#                                             elif i != 0 and l == 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             else:
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                                
#                                                 for k in range(i+1):
#                                                     P[i, l, l_b, 0, i - k, l - 1, l_b - 1] = arrival_probability(l, k)*P_b
#                                                 P[i, l, l_b, 0, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1)*P_b
                                                
#                                                 for k in range(i+1):
#                                                     P[i, l, l_b, 0, i - k, l - 1, l_b] = arrival_probability(l, k)*(1 - P_b)
#                                                 P[i, l, l_b, 0, 0, l - 1, l_b] = arr_greater_than(l, i - 1)*(1 - P_b)
#                             print('Probabilities completed')                   
#                             # Value iteration parameters
#                             # V = np.zeros((I+1,L+1, L+1))
#                             # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
#                             # # Value iteration
#                             # iteration = 0
#                             # while True:
#                             #     V_prev = np.copy(V)
#                             #     iteration += 1
#                             #     for i in range(I + 1):
#                             #         for l in range(L + 1):
#                             #             for l_b in range(L + 1):
#                             #                 if l >= l_b:
#                             #                     continue
#                             #                 else:
#                             #                     for a in range(num_actions):
#                             #                         if (i == 0 or l == 0) and a == 0:
#                             #                             V_actions[i, l, l_b, a] = -float('inf')
#                             #                         else:
#                             #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
#                             #     V = np.max(V_actions, axis = 3)
                                
#                             #     if np.max(np.abs(V - V_prev)) < epsilon:
#                             #         print('Value iteration completed')
#                             #         break
                                
#                             # optimal_policy = np.argmax(V_actions, axis = 3)

#                             def calculate_stationary_distribution(P, policy, I, L):
#                                 num_states = (I + 1) * (L + 1) * (L + 1)
#                                 P_flat = np.zeros((num_states, num_states))
                                
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                                             action = policy[i, l, l_b]
                                            
#                                             for i_next in range(I + 1):
#                                                 for l_next in range(L + 1):
#                                                     for l_b_next in range(L + 1):
#                                                         next_state_index = i_next * (L + 1) * (L + 1) + l_next * (L + 1) + l_b_next
#                                                         P_flat[state_index, next_state_index] = P[i, l, l_b, action, i_next, l_next, l_b_next]
                                
#                                 row_sums = P_flat.sum(axis=1)
                                
#                                 zero_sum_rows = (row_sums == 0)
#                                 P_flat[zero_sum_rows, :] = 1 / num_states
#                                 row_sums[zero_sum_rows] = 1
                                
#                                 P_flat = P_flat / row_sums[:, np.newaxis]
                                
#                                 if np.any(np.isnan(P_flat)) or np.any(np.isinf(P_flat)):
#                                     P_flat = np.nan_to_num(P_flat, nan=0, posinf=0, neginf=0)
#                                     row_sums = P_flat.sum(axis=1)
#                                     P_flat = P_flat / row_sums[:, np.newaxis]
                                
#                                 eigenvalues, eigenvectors = linalg.eig(P_flat.T)
#                                 index = np.argmin(np.abs(eigenvalues - 1))
#                                 stationary_dist = eigenvectors[:, index].real
#                                 stationary_dist = stationary_dist / np.sum(stationary_dist)
                                
#                                 stationary_dist_shaped = np.zeros((I + 1, L + 1, L + 1))
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                                             stationary_dist_shaped[i, l, l_b] = stationary_dist[state_index]
                                
#                                 return stationary_dist_shaped

#                             # Calculate optimal average reward
#                             # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
#                             # optimal_avg_reward = 0
#                             # for i in range(I+1):
#                             #     for l in range(L+1):
#                             #         for l_b in range(L+1):
#                             #             if l >= l_b:
#                             #                 continue
#                             #             else:
#                             #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]
#                             # print('Optimal average calculated')
#                             # print(optimal_avg_reward)
#                             # Calculate threshold policy results for i
#                             threshold_avg_results_i = {}
#                             for threshold in range(L + 1):
#                                 threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
#                                 threshold_policy[:threshold + 1, :, :] = 1
#                                 threshold_policy[:, 0, :] = 1
#                                 stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
#                                 threshold_avg_reward = 0
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             else:
#                                                 threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
#                                 print(f'Threshold calculated for i = {threshold}')
#                                 threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                                
#                             # # Calculate threshold policy results for l
#                             # threshold_avg_results_l = {}
#                             # for threshold in range(L + 1):
#                             #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
#                             #     threshold_policy[:, :threshold + 1, :] = 1
#                             #     threshold_policy[0, :, :] = 1
#                             #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
#                             #     threshold_avg_reward = 0
#                             #     for i in range(I + 1):
#                             #         for l in range(L + 1):
#                             #             for l_b in range(L + 1):
#                             #                 if l >= l_b:
#                             #                     continue
#                             #                 else:
#                             #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
#                             #     print(f'Threshold calculated for l = {threshold}')
#                             #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
#                             # Record results
#                             results.append({
#                                 "K": K,
#                                 "I": I,
#                                 "L": L,
#                                 "p": p,
#                                 "s": s,
#                                 "lambda_c": lambda_c,
#                                 "P_b": P_b,
#                                 # "Optimal Avg Reward": optimal_avg_reward,
#                                 **threshold_avg_results_i,
#                                 # **threshold_avg_results_l,
#                             })
#                             print(case)

# # Convert results to DataFrame and export to Excel
# df = pd.DataFrame(results)
# df.to_excel("mdp_simulation_results_with_threshold_K_with_missing.xlsx", index=False)

# # Define parameter ranges
# K_range = [90000]
# I_range = np.arange(80, 81, 1)
# L_range = [7]
# p_range = [1500]
# s_range = [400]
# lambda_c_range = [20]
# P_b_range = [0.1]

# num_actions = 2
# np.random.seed(42)

# # Prepare a DataFrame to store results
# results = []
# case = 0
# epsilon = 1e-6
# gamma = 0.99


# # Arrival probability functions
# def arrival_probability(L_i, k):
#     arr = lambda_c * (L_i / L)
#     return poisson.pmf(k, arr)

# def arr_greater_than(L_i, k):
#     arr = lambda_c * (L_i / L)
#     return 1 - poisson.cdf(k, arr)

# for K in K_range:
#     for I in I_range:
#         for L in L_range:
#             for p in p_range:
#                 for s in s_range:
#                     for lambda_c in lambda_c_range:
#                         for P_b in P_b_range:
#                             case += 1
                            
#                             # Reward array
#                             R = np.zeros((I + 1, L + 1, L + 1, num_actions))
#                             for i in range(I + 1):
#                                 for l in range(L + 1):
#                                     for l_b in range(L + 1):
#                                         for a in range(num_actions):
#                                             if a == 1:
#                                                 R[i, l, l_b, a] = - K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
#                                             else:
#                                                 R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p
#                             print('Rewards are calculated')
#                             # Transition probability matrix
#                             P = np.zeros((I + 1, L + 1, L + 1, num_actions, I + 1, L + 1, L + 1))
#                             for i in range(I + 1):
#                                 for l in range(L + 1):
#                                     for l_b in range(L + 1):
#                                         if l_b <= l and l != 0:
#                                             continue
#                                         else:
#                                             if i == 0 and l == 0 and l_b == 0:
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1
#                                                 P[i, l, l_b, 1, i, l, L] = 1
#                                             elif i == 0 and l == 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
#                                                 for k in range(I+1):    
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             elif i == 0 and l != 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l - 1, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l - 1, l_b] = 1 - P_b
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             elif i != 0 and l == 0 and l_b == 0:
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1
#                                                 P[i, l, l_b, 1, 0, l_b, L] = 1
#                                             elif i != 0 and l == 0 and l_b != 0:
#                                                 P[i, l, l_b, 0, i, l, l_b - 1] = P_b
#                                                 P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
#                                             else:
#                                                 for k in range(I+1):
#                                                     P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
#                                                 P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                                
#                                                 for k in range(i+1):
#                                                     P[i, l, l_b, 0, i - k, l - 1, l_b - 1] = arrival_probability(l, k)*P_b
#                                                 P[i, l, l_b, 0, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1)*P_b
                                                
#                                                 for k in range(i+1):
#                                                     P[i, l, l_b, 0, i - k, l - 1, l_b] = arrival_probability(l, k)*(1 - P_b)
#                                                 P[i, l, l_b, 0, 0, l - 1, l_b] = arr_greater_than(l, i - 1)*(1 - P_b)
#                             print('Probabilities are calculated')                   
#                             # # Value iteration parameters
#                             # V = np.zeros((I+1,L+1, L+1))
#                             # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
#                             # # Value iteration
#                             # iteration = 0
#                             # while True:
#                             #     V_prev = np.copy(V)
#                             #     iteration += 1
#                             #     for i in range(I + 1):
#                             #         for l in range(L + 1):
#                             #             for l_b in range(L + 1):
#                             #                 if l >= l_b:
#                             #                     continue
#                             #                 else:
#                             #                     for a in range(num_actions):
#                             #                         if (i == 0 or l == 0) and a == 0:
#                             #                             V_actions[i, l, l_b, a] = -float('inf')
#                             #                         else:
#                             #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
#                             #     V = np.max(V_actions, axis = 3)
#                             #     if np.max(np.abs(V - V_prev)) < epsilon:
#                             #         print('Value iteration converged')
#                             #         break
                                
#                             # optimal_policy = np.argmax(V_actions, axis = 3)

#                             def calculate_stationary_distribution(P, policy, I, L):
#                                 num_states = (I + 1) * (L + 1) * (L + 1)
#                                 P_flat = np.zeros((num_states, num_states))
                                
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                                             action = policy[i, l, l_b]
                                            
#                                             for i_next in range(I + 1):
#                                                 for l_next in range(L + 1):
#                                                     for l_b_next in range(L + 1):
#                                                         next_state_index = i_next * (L + 1) * (L + 1) + l_next * (L + 1) + l_b_next
#                                                         P_flat[state_index, next_state_index] = P[i, l, l_b, action, i_next, l_next, l_b_next]
                                
#                                 row_sums = P_flat.sum(axis=1)
                                
#                                 zero_sum_rows = (row_sums == 0)
#                                 P_flat[zero_sum_rows, :] = 1 / num_states
#                                 row_sums[zero_sum_rows] = 1
                                
#                                 P_flat = P_flat / row_sums[:, np.newaxis]
                                
#                                 if np.any(np.isnan(P_flat)) or np.any(np.isinf(P_flat)):
#                                     P_flat = np.nan_to_num(P_flat, nan=0, posinf=0, neginf=0)
#                                     row_sums = P_flat.sum(axis=1)
#                                     P_flat = P_flat / row_sums[:, np.newaxis]
                                
#                                 eigenvalues, eigenvectors = linalg.eig(P_flat.T)
#                                 index = np.argmin(np.abs(eigenvalues - 1))
#                                 stationary_dist = eigenvectors[:, index].real
#                                 stationary_dist = stationary_dist / np.sum(stationary_dist)
                                
#                                 stationary_dist_shaped = np.zeros((I + 1, L + 1, L + 1))
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                                             stationary_dist_shaped[i, l, l_b] = stationary_dist[state_index]
                                
#                                 return stationary_dist_shaped

#                             # Calculate optimal average reward
#                             # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
#                             # optimal_avg_reward = 0
#                             # for i in range(I+1):
#                             #     for l in range(L+1):
#                             #         for l_b in range(L+1):
#                             #             if l >= l_b:
#                             #                 continue
#                             #             else:
#                             #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]
#                             # print('Optimal average calculated')
#                             # Calculate threshold policy results for i
#                             threshold_avg_results_i = {}
#                             for threshold in range(L + 1):
#                                 threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
#                                 threshold_policy[:threshold + 1, :, :] = 1
#                                 threshold_policy[:, 0, :] = 1
#                                 stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
#                                 threshold_avg_reward = 0
#                                 for i in range(I + 1):
#                                     for l in range(L + 1):
#                                         for l_b in range(L + 1):
#                                             if l >= l_b:
#                                                 continue
#                                             else:
#                                                 threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
#                                 threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
#                                 print(f'Threshold calculated for {threshold}')
#                             # Calculate threshold policy results for l
#                             # threshold_avg_results_l = {}
#                             # for threshold in range(L + 1):
#                             #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
#                             #     threshold_policy[:, :threshold + 1, :] = 1
#                             #     threshold_policy[0, :, :] = 1
#                             #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
#                             #     threshold_avg_reward = 0
#                             #     for i in range(I + 1):
#                             #         for l in range(L + 1):
#                             #             for l_b in range(L + 1):
#                             #                 if l >= l_b:
#                             #                     continue
#                             #                 else:
#                             #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
#                             #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
#                             #     print(f'Threshold calculated for {threshold}')
#                             # Record results
#                             results.append({
#                                 "K": K,
#                                 "I": I,
#                                 "L": L,
#                                 "p": p,
#                                 "s": s,
#                                 "lambda_c": lambda_c,
#                                 "P_b": P_b,
#                                 # "Optimal Avg Reward": optimal_avg_reward,
#                                 **threshold_avg_results_i,
#                                 # **threshold_avg_results_l,
#                             })
#                             print(case)

# # Convert results to DataFrame and export to Excel
# df = pd.DataFrame(results)
# df.to_excel("mdp_simulation_results_with_threshold_I_with_missing.xlsx", index=False)

# Define parameter ranges
K_range = [30000]
I_range = [30]
L_range = np.arange(2, 13, 1)
p_range = [1500]
s_range = [400]
lambda_c_range = [10]
P_b_range = [0.1]

num_actions = 2
np.random.seed(42)

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
                                                
                            # # Value iteration parameters
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # # Value iteration
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration += 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     for a in range(num_actions):
                            #                         if (i == 0 or l == 0) and a == 0:
                            #                             V_actions[i, l, l_b, a] = -float('inf')
                            #                         else:
                            #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # optimal_policy = np.argmax(V_actions, axis = 3)

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

                            # # Calculate optimal average reward
                            # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            # optimal_avg_reward = 0
                            # for i in range(I+1):
                            #     for l in range(L+1):
                            #         for l_b in range(L+1):
                            #             if l >= l_b:
                            #                 continue
                            #             else:
                            #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

                            # # Calculate threshold policy results for i
                            threshold_avg_results_i = {}
                            for threshold in range(L + 1):
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                            # Calculate threshold policy results for l
                            # threshold_avg_results_l = {}
                            # for threshold in range(L + 1):
                            #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                            #     threshold_policy[:, :threshold + 1, :] = 1
                            #     threshold_policy[0, :, :] = 1
                            #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                            #     threshold_avg_reward = 0
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                            #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                # "Optimal Avg Reward": optimal_avg_reward,
                                **threshold_avg_results_i,
                                # **threshold_avg_results_l,
                            })
                            print(case)

# Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("mdp_simulation_results_with_threshold_L_with_missing.xlsx", index=False)

# Define parameter ranges


K_range = [30000]
I_range = [30]
L_range = [4]
p_range = np.arange(1200, 1801, 20)
s_range = [400]
lambda_c_range = [10]
P_b_range = [0.1]
num_actions = 2
np.random.seed(42)

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
                                                
                            # Value iteration parameters
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # # Value iteration
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration += 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     for a in range(num_actions):
                            #                         if (i == 0 or l == 0) and a == 0:
                            #                             V_actions[i, l, l_b, a] = -float('inf')
                            #                         else:
                            #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # optimal_policy = np.argmax(V_actions, axis = 3)

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
                            # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            # optimal_avg_reward = 0
                            # for i in range(I+1):
                            #     for l in range(L+1):
                            #         for l_b in range(L+1):
                            #             if l >= l_b:
                            #                 continue
                            #             else:
                            #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

                            # Calculate threshold policy results for i
                            threshold_avg_results_i = {}
                            for threshold in range(L + 1):
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                            # Calculate threshold policy results for l
                            # threshold_avg_results_l = {}
                            # for threshold in range(L + 1):
                            #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                            #     threshold_policy[:, :threshold + 1, :] = 1
                            #     threshold_policy[0, :, :] = 1
                            #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                            #     threshold_avg_reward = 0
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                            #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                # "Optimal Avg Reward": optimal_avg_reward,
                                **threshold_avg_results_i,
                                # **threshold_avg_results_l,
                            })
                            print(case)

# Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("mdp_simulation_results_with_threshold_p_with_missing.xlsx", index=False)

# Define parameter ranges


K_range = [30000]
I_range = [30]
L_range = [4]
p_range = [1500]
s_range = np.arange(300, 500, 10)
lambda_c_range = [10]
P_b_range = [0.1]
num_actions = 2
np.random.seed(42)

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
                                                
                            # Value iteration parameters
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # # Value iteration
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration += 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     for a in range(num_actions):
                            #                         if (i == 0 or l == 0) and a == 0:
                            #                             V_actions[i, l, l_b, a] = -float('inf')
                            #                         else:
                            #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # optimal_policy = np.argmax(V_actions, axis = 3)

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
                            # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            # optimal_avg_reward = 0
                            # for i in range(I+1):
                            #     for l in range(L+1):
                            #         for l_b in range(L+1):
                            #             if l >= l_b:
                            #                 continue
                            #             else:
                            #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

                            # Calculate threshold policy results for i
                            threshold_avg_results_i = {}
                            for threshold in range(L + 1):
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                            # Calculate threshold policy results for l
                            # threshold_avg_results_l = {}
                            # for threshold in range(L + 1):
                            #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                            #     threshold_policy[:, :threshold + 1, :] = 1
                            #     threshold_policy[0, :, :] = 1
                            #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                            #     threshold_avg_reward = 0
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                            #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                # "Optimal Avg Reward": optimal_avg_reward,
                                **threshold_avg_results_i,
                                # **threshold_avg_results_l,
                            })
                            print(case)

# Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("mdp_simulation_results_with_threshold_s_with_missing.xlsx", index=False)


K_range = [30000]
I_range = [30]
L_range = [4]
p_range = [1500]
s_range = [400]
lambda_c_range = np.arange(5,21,1)
P_b_range = [0.1]
num_actions = 2
np.random.seed(42)

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
                                                
                            # Value iteration parameters
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # Value iteration
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration += 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     for a in range(num_actions):
                            #                         if (i == 0 or l == 0) and a == 0:
                            #                             V_actions[i, l, l_b, a] = -float('inf')
                            #                         else:
                            #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # optimal_policy = np.argmax(V_actions, axis = 3)

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
                            # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            # optimal_avg_reward = 0
                            # for i in range(I+1):
                            #     for l in range(L+1):
                            #         for l_b in range(L+1):
                            #             if l >= l_b:
                            #                 continue
                            #             else:
                            #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

                            # # Calculate threshold policy results for i
                            threshold_avg_results_i = {}
                            for threshold in range(L + 1):
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                            # Calculate threshold policy results for l
                            # threshold_avg_results_l = {}
                            # for threshold in range(L + 1):
                            #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                            #     threshold_policy[:, :threshold + 1, :] = 1
                            #     threshold_policy[0, :, :] = 1
                            #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                            #     threshold_avg_reward = 0
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                # threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                # "Optimal Avg Reward": optimal_avg_reward,
                                **threshold_avg_results_i,
                                # **threshold_avg_results_l,
                            })
                            print(case)

# Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("mdp_simulation_results_with_threshold_lambda_c_with_missing.xlsx", index=False)


K_range = [30000]
I_range = [30]
L_range = [4]
p_range = [1500]
s_range = [400]
lambda_c_range = [10]
P_b_range = np.arange(0, 1, 0.1)
num_actions = 2
np.random.seed(42)

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
                                                
                            # Value iteration parameters
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # # Value iteration
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration += 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     for a in range(num_actions):
                            #                         if (i == 0 or l == 0) and a == 0:
                            #                             V_actions[i, l, l_b, a] = -float('inf')
                            #                         else:
                            #                             V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                                
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # optimal_policy = np.argmax(V_actions, axis = 3)

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
                            # stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)
                            
                            # optimal_avg_reward = 0
                            # for i in range(I+1):
                            #     for l in range(L+1):
                            #         for l_b in range(L+1):
                            #             if l >= l_b:
                            #                 continue
                            #             else:
                            #                 optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

                            # Calculate threshold policy results for i
                            threshold_avg_results_i = {}
                            for threshold in range(L + 1):
                                threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                                threshold_policy[:threshold + 1, :, :] = 1
                                threshold_policy[:, 0, :] = 1
                                stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                                threshold_avg_reward = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l >= l_b:
                                                continue
                                            else:
                                                threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                                threshold_avg_results_i[f"Threshold {threshold} Avg"] = threshold_avg_reward
                            # Calculate threshold policy results for l
                            # threshold_avg_results_l = {}
                            # for threshold in range(L + 1):
                            #     threshold_policy = np.zeros((I + 1, L + 1, L + 1), dtype=int)
                            #     threshold_policy[:, :threshold + 1, :] = 1
                            #     threshold_policy[0, :, :] = 1
                            #     stationary_dist_threshold = calculate_stationary_distribution(P, threshold_policy, I, L)
                                
                            #     threshold_avg_reward = 0
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 if l >= l_b:
                            #                     continue
                            #                 else:
                            #                     threshold_avg_reward += R[i, l, l_b, threshold_policy[i, l, l_b]] * stationary_dist_threshold[i, l, l_b]
                                
                            #     threshold_avg_results_l[f"Threshold {threshold} Avg Lifetime"] = threshold_avg_reward
                            
                            # Record results
                            results.append({
                                "K": K,
                                "I": I,
                                "L": L,
                                "p": p,
                                "s": s,
                                "lambda_c": lambda_c,
                                "P_b": P_b,
                                # "Optimal Avg Reward": optimal_avg_reward,
                                **threshold_avg_results_i,
                                # **threshold_avg_results_l,
                            })
                            print(case)

# Convert results to DataFrame and export to Excel
df = pd.DataFrame(results)
df.to_excel("mdp_simulation_results_with_threshold_P_b_with_missing.xlsx", index=False)
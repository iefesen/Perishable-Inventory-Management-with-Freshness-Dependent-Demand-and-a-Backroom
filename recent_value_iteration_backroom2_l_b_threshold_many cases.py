# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:23:16 2024

@author: 20224695
"""

import numpy as np
from scipy.stats import poisson
import pandas as pd



K_range = [35, 50]
I_range = [20, 40]
L_range = [8, 12]
p_range = [6, 10]
s_range = [0.4, 1]
lambda_c_range = [12, 16]
P_b_range = [0.3, 0.7]

num_actions = 2

# K_range = [10, 20]
# I_range = [10]
# L_range = [5]
# p_range = [4]
# s_range = [0.1]
# lambda_c_range = [5]
# P_b_range = [0.1]

results = []

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

'The function providing the arrival probability of k customers within a period for a given lifetime level of L_i'
def arrival_probability(L_i, k):
    arr = lambda_c*(L_i/L)
    return poisson.pmf(k, arr)


'The function providing the arrival probability of more than k customers for a given lifetime level L_i'        
def arr_greater_than(L_i, k):
    arr = lambda_c*(L_i/L)
    return 1 - poisson.cdf(k, arr)

def compute_steady_state_distribution(transition_matrix):
    num_states = (I+1)*(L+1)*(L+1)
    
    # Construct the matrix A for the equation A * pi = b
    A = np.eye(num_states) - transition_matrix.T
    A[-1, :] = 1  # Replace the last row with 1s to ensure a valid probability distribution
    
    b = np.zeros(num_states)
    b[-1] = 1  # The sum of probabilities is 1
    
    try:
        steady_state_distribution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        print("Linear algebra error:", e)
        steady_state_distribution = None
    
    return steady_state_distribution



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
case = 0

for K in K_range:
    for I in I_range:
        for L in L_range:
            for p in p_range:
                for s in s_range:
                    for lambda_c in lambda_c_range:
                        for P_b in P_b_range:
                            case = case + 1
                            # Reward array
                            R = np.zeros((I + 1, L + 1, L + 1, num_actions))

                            for i in range(I + 1):
                                for l in range(L + 1):
                                    for l_b in range(L + 1):
                                        if l > l_b:
                                            continue
                                        for a in range(num_actions):
                                            if a == 1 and l_b != 0:
                                                R[i, l, l_b, a] = - K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
                                            elif a == 1 and l_b == 0:
                                                R[i, l, l_b, a] = - 2 * K + s * (I + i) + p * (sum(k * arrival_probability(L, k) for k in range(I))) + arr_greater_than(L, I - 1) * I * p
                                            else:
                                                R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p


                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################


                            # Transition probabilities array
                            P = np.zeros((I + 1, L + 1, L + 1, num_actions, I + 1, L + 1, L + 1))

                            for i in range(I + 1):
                                for l in range(L + 1):
                                    for l_b in range(L + 1):
                                        if l > l_b:
                                            continue
                                        for a in range(num_actions):
                                            if a == 1 and l_b != 0:
                                                P[i, l, l_b, a, 0, l_b - 1, L - 1] = arr_greater_than(l_b, I - 1) * P_b
                                                for k in range(I):
                                                    P[i, l, l_b, a, I - k, l_b - 1, L - 1] = arrival_probability(l_b, k) * P_b
                                                P[i, l, l_b, a, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1) * (1 - P_b)
                                                for k in range(I):
                                                    P[i, l, l_b, a, I - k, l_b - 1, L] = arrival_probability(l_b, k) * (1 - P_b)
                                            elif a == 1 and l_b == 0:
                                                P[i, l, l_b, a, 0, L - 1, L - 1] = arr_greater_than(L, I - 1) * P_b
                                                for k in range(I):
                                                    P[i, l, l_b, a, I - k, L - 1, L - 1] = arrival_probability(L, k) * P_b
                                                P[i, l, l_b, a, 0, L - 1, L] = arr_greater_than(L, I - 1) * (1 - P_b)
                                                for k in range(I):
                                                    P[i, l, l_b, a, I - k, L - 1, L] = arrival_probability(L, k) * (1 - P_b)
                                            else:
                                                if l > 0:  
                                                    P[i, l, l_b, a, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1) * P_b
                                                    for k in range(i):
                                                        P[i, l, l_b, a, i - k, l - 1, l_b - 1] = arrival_probability(l, k) * P_b
                                                    P[i, l, l_b, a, 0, l - 1, l_b] = arr_greater_than(l, i - 1) * (1 - P_b)
                                                    for k in range(i):
                                                        P[i, l, l_b, a, i - k, l - 1, l_b] = arrival_probability(l, k) * (1 - P_b)
                                                elif l_b > 0:
                                                    P[i, l, l_b, a, i, l, l_b - 1] = P_b
                                                    P[i, l, l_b, a, i, l, l_b] = 1 - P_b
                                                else:
                                                    P[i, l, l_b, a, i, l, l_b] = 1
                                                     

                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            # 'Value iteration parameters'
                            # epsilon = 1e-6
                            # gamma = 0.99

                            # 'Empty sets containing the value of each state'
                            # V = np.zeros((I+1,L+1, L+1))
                            # V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))

                            # 'Value iteration'
                            # iteration = 0
                            # while True:
                            #     V_prev = np.copy(V)
                            #     iteration = iteration + 1
                            #     for i in range(I + 1):
                            #         for l in range(L + 1):
                            #             for l_b in range(L + 1):
                            #                 for a in range(num_actions):
                            #                     V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                                        
                            #     V = np.max(V_actions, axis = 3)
                            #     if np.max(np.abs(V - V_prev)) < epsilon:
                            #         break
                                
                            # 'Finding the optimal policy'
                            # optimal_policy = np.argmax(V_actions, axis = 3)

                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            # 'We omit the optimal policy analysis as the i threshold analysis already contains it'
                            # 'Average reward analysis'
                            
                            # # Create the transition matrix for the optimal policy
                            # P_opt = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))

                            # for i in range(I + 1):
                            #     for l in range(L + 1):
                            #         for l_b in range(L + 1):
                            #             P_opt[i, l, l_b] = P[i, l, l_b, optimal_policy[i, l, l_b]]

                            # P_opt_flat = P_opt.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))
                            # steady_state_opt = compute_steady_state_distribution(P_opt_flat)
                            # # For the optimal policy
                            # P_opt = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))

                            # for i in range(I + 1):
                            #     for l in range(L + 1):
                            #         for l_b in range(L + 1):
                            #             P_opt[i, l, l_b] = P[i, l, l_b, optimal_policy[i, l, l_b]]

                            # P_opt_flat = P_opt.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))

                            # steady_state_opt = compute_steady_state_distribution(P_opt_flat)
                            # steady_state_opt = steady_state_opt.reshape((I + 1, L + 1, L + 1))

                            # opt_average = 0

                            # for i in range(I + 1):
                            #     for l in range(L + 1):
                            #         for l_b in range(L + 1):
                            #             opt_average += R[i, l, l_b, optimal_policy[i, l, l_b]] * steady_state_opt[i, l, l_b]
                                        
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            averages = []
                            
                            for l_b_t in range(L + 1):
                                P_threshold = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l_b < l_b_t or l == 0:
                                                P_threshold[i, l, l_b] = P[i, l, l_b, 1]
                                            else:
                                                P_threshold[i, l, l_b] = P[i, l, l_b, 0]

                                P_threshold_flat = P_threshold.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))

                                steady_state = compute_steady_state_distribution(P_threshold_flat)
                                steady_state = steady_state.reshape((I + 1, L + 1, L + 1))

                                average = 0
                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            if l_b < l_b_t or i == 0 or l == 0:
                                                average += R[i, l, l_b, 1] * steady_state[i, l, l_b]
                                            else:
                                                average += R[i, l, l_b, 0] * steady_state[i, l, l_b]
                                averages.append(average)
                            
                            row = [K, I, L, p, s, lambda_c, P_b] + averages
                            
                            results.append(row)
                            print(case)


columns = ['K', 'I', 'L', 'p', 's', 'lambda_c', 'P_b'] + [f'average_{l_b_t}' for l_b_t in range(L + 1)]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_excel('average_rewards_threshold_l_b_analysis2.xlsx', index=False)
        

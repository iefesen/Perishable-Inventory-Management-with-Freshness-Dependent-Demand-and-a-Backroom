# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 15:57:51 2024

@author: 20224695
"""

import numpy as np
from scipy.stats import poisson
import pandas as pd

# Parameters

K_range = [35, 50]
I_range = [20, 40]
L_range = [8, 12]
p_range = [6, 10]
s_range = [0.4, 1]
lambda_c_range = [12, 16]
P_b_range = [0.3, 0.7]

num_actions = 2

np.random.seed(42)
results = []
case = 0
for K in K_range:
    for I in I_range:
        for L in L_range:
            for p in p_range:
                for s in s_range:
                    for lambda_c in lambda_c_range:
                        for P_b in P_b_range:
                            case = case + 1
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            # Arrival probability functions
                            def arrival_probability(L_i, k):
                                arr = lambda_c * (L_i / L)
                                return poisson.pmf(k, arr)
                            
                            def arr_greater_than(L_i, k):
                                arr = lambda_c * (L_i / L)
                                return 1 - poisson.cdf(k, arr)
                            
                            
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            
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
                            
                            
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            'Transition probability matrix'
                            
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
                            
                                                
                            
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            'Value iteration parameters'
                            epsilon = 1e-6
                            gamma = 0.99
                            
                            'Empty sets containing the value of each state'
                            V = np.zeros((I+1,L+1, L+1))
                            V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            'Value iteration'
                            iteration = 0
                            while True:
                                V_prev = np.copy(V)
                                iteration = iteration + 1
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
                                    # print('The value iteration converged at iteration', iteration)
                                    break
                                
                            'Finding the optimal policy'
                            optimal_policy = np.argmax(V_actions, axis = 3)
                            
                            
                            
                            
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            #############################################################################################
                            
                            def simulate_mdp(policy, P, R, start_state, num_steps):
                                state = start_state
                                rewards_obtained = 0
                                per_time_rewards_obtained = float('inf')
                                for step in range(num_steps):
                                    i, l, l_b = state
                                    action = policy[i, l, l_b]
                                    
                                    # Get the reward for the current state-action pair
                                    reward = R[i, l, l_b, action]
                                    rewards_obtained = rewards_obtained + reward
                                    
                                    # Get the transition probabilities for the current state-action pair
                                    transition_probs = P[i, l, l_b, action]
                                    
                                    # Flatten the transition probabilities to select the next state
                                    transition_probs_flat = transition_probs.flatten()
                                    next_state_index = np.random.choice(np.arange(transition_probs_flat.size), p=transition_probs_flat)
                                    
                                    # Convert flat index back to state indices
                                    next_state = np.unravel_index(next_state_index, (I+1, L+1, L+1))
                                    
                                    # Update state
                                    state = next_state
                                    per_time_rewards_obtained_new = rewards_obtained / (step + 1)
                                    if abs(per_time_rewards_obtained - per_time_rewards_obtained_new) <1e-4 and step > 1000:
                                        per_time_rewards_obtained = per_time_rewards_obtained_new
                                        break
                                    else:
                                        per_time_rewards_obtained = per_time_rewards_obtained_new
                                return per_time_rewards_obtained
                            
                            # Parameters
                            max_simulations = 1000  # Maximum number of simulations
                            num_steps = 100000  # Maximum number of steps per simulation
                            start_state = (0, 0, L)  # Starting state (you can change this)
                            
                            # Run the simulation
                            total_of_sims = 0
                            avg_of_sims = float('inf')
                            per_time_rewards_obtained_sims = []
                            for sim in range(max_simulations):
                                per_time_rewards_obtained = simulate_mdp(optimal_policy, P, R, start_state, num_steps)
                                per_time_rewards_obtained_sims.append(per_time_rewards_obtained)
                                total_of_sims = total_of_sims + per_time_rewards_obtained
                                avg_of_sims_new = total_of_sims / (sim + 1)
                                if abs(avg_of_sims - avg_of_sims_new) < 1e-4:
                                    avg_of_sims = avg_of_sims_new
                                    # print('Average reward is: ', avg_of_sims)
                                    # print('Converged at simulation: ', sim + 1)
                                    break
                                else:
                                    # print('The difference is: ', abs(avg_of_sims - avg_of_sims_new))
                                    avg_of_sims = avg_of_sims_new
                            
                            row = [K, I, L, p, s, lambda_c, P_b] + per_time_rewards_obtained_sims
                            results.append(row)        
                            print(case)
                            
                            

results_df = pd.DataFrame(results)
results_df.to_excel('sim_optimal.xlsx', index=False)





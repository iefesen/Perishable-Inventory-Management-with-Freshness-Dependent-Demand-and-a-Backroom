# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:05:55 2025

@author: 20224695
"""

import numpy as np
from scipy.stats import poisson
from scipy import linalg
import matplotlib.pyplot as plt


num_actions = 2

epsilon = 1e-6
gamma = 0.99
max_iterations = 10000

# Arrival probability functions
def arrival_probability(L_i, k):
    arr = lambda_c * (L_i / L)
    return poisson.pmf(k, arr)

def arr_greater_than(L_i, k):
    arr = lambda_c * (L_i / L)
    return 1 - poisson.cdf(k, arr)

def calculate_stationary_distribution(P, policy, I, L):
    num_states = (2*I + 1) * (L + 1) * (L + 1)
    P_flat = np.zeros((num_states, num_states))
    
    for i in range(2*I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if l >= l_b:
                    continue
                state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
                action = policy[i, l, l_b]
                
                for i_next in range(2*I + 1):
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
    
    stationary_dist_shaped = np.zeros((2*I + 1, L + 1, L + 1))
    for i in range(2*I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if l >= l_b:
                    continue
                state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
                stationary_dist_shaped[i, l, l_b] = stationary_dist[state_index]
    
    return stationary_dist_shaped
# Value Iteration Algorithm

def value_iteration(R, P, gamma, epsilon, max_iterations):
    
    V = np.zeros((2*I + 1, L + 1, L + 1))  # Initialize value function
    policy = np.zeros((2*I + 1, L + 1, L + 1), dtype=int)  # Initialize policy
    action_values_store = np.full((2*I + 1, L + 1, L + 1, num_actions), -np.inf)  # Default infeasible actions to -∞

    for iteration in range(max_iterations):
        delta = 0
        new_V = np.copy(V)

        for i in range(2*I + 1):
            for l in range(L + 1):
                for l_b in range(L + 1):
                    if l > l_b:
                        continue  # Skip invalid states

                    action_values = np.full(num_actions, -np.inf)  # Default infeasible actions to -∞

                    for a in range(num_actions):
                        # **Force action 1 when i == 0 or l == 0**
                        if (i == 0 or l == 0) and a == 0:
                            continue  # Prevents choosing action 0 in these states

                        # **Force action 0 when i >= I + 1**
                        if i >= I + 1 and a == 1:
                            continue  # Prevents choosing action 1 when i ≥ I+1

                        value_sum = 0
                        for i_next in range(2*I + 1):
                            for l_next in range(L + 1):
                                for l_b_next in range(L + 1):
                                    prob = P[i, l, l_b, a, i_next, l_next, l_b_next]
                                    value_sum += prob * (R[i, l, l_b, a] + gamma * V[i_next, l_next, l_b_next])

                        action_values[a] = value_sum  # Store valid action value

                    action_values_store[i, l, l_b] = action_values  # Store all action values for analysis
                    best_action = np.argmax(action_values)  # Pick the action with the highest value
                    new_V[i, l, l_b] = action_values[best_action]  # Update V
                    policy[i, l, l_b] = best_action  # Store best action
                    delta = max(delta, abs(new_V[i, l, l_b] - V[i, l, l_b]))  # Track convergence

        V = new_V
        if delta < epsilon:
            break  # Converged

    return V, policy, action_values_store


# Plot for l_b
def plot_optimal_policy(optimal_policy, I, L):
    for l_b in range(L+1):
        l_b_fixed = l_b  # Adjust as needed
        
        # Extract points where optimal_policy is 1
        points_x = []
        points_y = []
        
        for i in range(I + 1):
            for l in range(L + 1):
                if optimal_policy[i, l, l_b_fixed] == 1:
                    points_x.append(i)
                    points_y.append(l)
        
        # Create scatter plot with discrete values
        plt.figure(figsize=(8, 6))
        plt.scatter(points_x, points_y, marker='o', color='b', alpha=0.7)
        plt.xticks(range(I + 1))
        plt.yticks(range(L + 1))
        plt.xlabel("Inventory Level (i)")
        plt.ylabel("Shelf Life (l)")
        plt.title(f"Scatter Plot of Optimal Policy = 1 (l_b={l_b_fixed})")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()



def plot_action_values(I, L, l_fixed, l_b_fixed, action_values_store, num_actions):
    if l_b_fixed < l_fixed:
        # print("Infeasible state: l_b must be at least l.")
        return

    plt.figure(figsize=(8, 6))

    for a in range(num_actions):
        valid_indices = [i for i in range(2*I + 1) if not (a == 1 and i > I)]  # Exclude replenishing when i > I
        values = [action_values_store[i, l_fixed, l_b_fixed, a] for i in valid_indices]

        plt.plot(valid_indices, values, linestyle='-', label=f"Action {a}")

    plt.xlabel("Inventory Level (i)")
    plt.ylabel("Value")
    plt.title(f"Value of Each Action (l={l_fixed}, l_b={l_b_fixed})")
    plt.legend()
    plt.grid(True)
    plt.show()


# Full factorial parameters
K_range = [10]
I_range = [10,12,14,16]
L_range = [3,4,5,6]
lambda_c_range = [8]
p_range = [15]
P_b_range = [0.1]

for K in K_range:
    for I in I_range:
        for L in L_range:
            for lambda_c in lambda_c_range:
                for p in p_range:
                    for P_b in P_b_range:
                        # Reward array
                        R = np.zeros((2*I + 1, L + 1, L + 1, num_actions))
                        for i in range(2*I + 1):
                            for l in range(L + 1):
                                for l_b in range(L + 1):
                                    for a in range(num_actions):
                                        if a == 1 and i != 0 and l != 0:
                                            R[i, l, l_b, a] = - K + p * (sum(k * arrival_probability(round((l_b + l)/(2)), k) for k in range(I + i))) + arr_greater_than(round((l_b + l)/(2)), I + i - 1) * (I + i) * p
                                        elif a == 1 and (i == 0 or l == 0):
                                            R[i, l, l_b, a] = - K + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * (I) * p                    
                                        else:
                                            R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p
                        
                        print('Rewards completed')
                        
                        P = np.zeros((2*I + 1, L + 1, L + 1, num_actions, 2*I + 1, L + 1, L + 1))
                        for i in range(2*I + 1):
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
                                            P[i, l, l_b, 1, I, l_b, L] = 1
                                        elif i != 0 and l == 0 and l_b != 0:
                                            P[i, l, l_b, 0, i, l, l_b - 1] = P_b
                                            P[i, l, l_b, 0, i, l, l_b] = 1 - P_b
                                            for k in range(I+1):
                                                P[i, l, l_b, 1, I - k, l_b - 1, L] = arrival_probability(l_b, k)
                                            P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(l_b, I - 1)
                                        elif i < I:
                                            for k in range(I + i + 1):
                                                P[i, l, l_b, 1, I + i - k, round((l_b + l)/(2)) - 1, L] = arrival_probability(round((l_b + l)/(2)), k)
                                            P[i, l, l_b, 1, 0, l_b - 1, L] = arr_greater_than(round((l_b + l)/(2)), I + i - 1)
                        
                                            for k in range(i+1):
                                                P[i, l, l_b, 0, i - k, l - 1, l_b - 1] = arrival_probability(l, k)*P_b
                                            P[i, l, l_b, 0, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1)*P_b
                                            
                                            for k in range(i+1):
                                                P[i, l, l_b, 0, i - k, l - 1, l_b] = arrival_probability(l, k)*(1 - P_b)
                                            P[i, l, l_b, 0, 0, l - 1, l_b] = arr_greater_than(l, i - 1)*(1 - P_b)
                                        else:
                                            for k in range(i+1):
                                                P[i, l, l_b, 0, i - k, l - 1, l_b - 1] = arrival_probability(l, k)*P_b
                                            P[i, l, l_b, 0, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1)*P_b
                                            
                                            for k in range(i+1):
                                                P[i, l, l_b, 0, i - k, l - 1, l_b] = arrival_probability(l, k)*(1 - P_b)
                                            P[i, l, l_b, 0, 0, l - 1, l_b] = arr_greater_than(l, i - 1)*(1 - P_b)
                        
                        print('Probabilities completed')                   
                        
                        optimal_values, optimal_policy, action_values_store = value_iteration(R, P, gamma, epsilon, max_iterations)
                        plot_optimal_policy(optimal_policy, I, L)
                        for l in range(L+1):
                            for l_b in range(L+1):
                                l_fixed = l
                                l_b_fixed = l_b
                                plot_action_values(I, L, l_fixed, l_b_fixed, action_values_store, num_actions)




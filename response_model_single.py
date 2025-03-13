import numpy as np
from scipy.stats import poisson
from scipy import linalg
import matplotlib.pyplot as plt
import time

start_time = time.time()
num_actions = 2
epsilon = 1e-6
gamma = 0.99

# Parameters
K = 240
I = 26
L = 7
p = 15
lambda_c = 16
P_b = 0


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

# Reward array
R = np.zeros((2*I + 1, L + 1, L + 1, num_actions))
for i in range(2*I + 1):
    for l in range(L + 1):
        for l_b in range(L + 1):
            for a in range(num_actions):
                if a == 1 and i != 0 and l != 0:
                    R[i, l, l_b, a] = - K + p * (sum(k * arrival_probability(round((I*l_b + i*l)/(I+i)), k) for k in range(I + i))) + arr_greater_than(round((I*l_b + i*l)/(I+i)), I + i - 1) * (I + i) * p
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
                if (i > I and a == 1) or ((i == 0 or l == 0) and a == 0):
                    P[i, l, l_b, a] = 0  # explicitly no transitions allowed
                elif i == 0 and l == 0 and l_b == 0:
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
                elif i <= I:
                    new_l = round((I*l_b + i*l)/(I+i))
                    for k in range(I + i + 1):
                        P[i, l, l_b, 1, I + i - k, new_l - 1, L] = arrival_probability(new_l, k)
                    P[i, l, l_b, 1, 0, new_l - 1, L] = arr_greater_than(new_l, I + i - 1)

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
                    
for i in range(2*I+1):
    for l in range(L+1):
        for l_b in range(L+1):
            if l >= l_b or (i > I):
                continue
            else: 
                for a in range(num_actions):
                    sum_ = 0
                    for i_ in range(2*I+1):
                        for l_ in range(L+1):
                            for l_b_ in range(L+1):
                                sum_ = sum_ + P[i, l, l_b, a, i_, l_, l_b_]
                if abs(sum_ - 1) > epsilon:
                    print(f'Problem at state {i}, {l}, {l_b} under action {a} probability {sum_}')
                                
print('Probabilities completed')                   
    
V = np.zeros((2*I+1,L+1, L+1))
V_actions = np.zeros((2*I + 1, L + 1, L + 1 , num_actions))

# Value iteration
iteration = 0
while True:
    V_prev = np.copy(V)
    iteration += 1
    for i in range(2*I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if l >= l_b:
                    continue
                else:
                    for a in range(num_actions):
                        if ((i == 0 or l == 0) and a == 0): 
                            V_actions[i, l, l_b, a] = -float('inf')
                        elif i > I and a == 1 and l != 0:
                            V_actions[i, l, l_b, a] = -float('inf')
                        else:
                            V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
                    
    V = np.max(V_actions, axis = 3)
    
    if np.max(np.abs(V - V_prev)) < epsilon:
        print('Value iteration completed at iteration: ', iteration)
        break
    
optimal_policy = np.argmax(V_actions, axis = 3)
end_time = time.time()

stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)

optimal_avg_reward = 0
for i in range(2*I+1):
    for l in range(L+1):
        for l_b in range(L+1):
            if l >= l_b:
                continue
            else:
                optimal_avg_reward += R[i, l, l_b, optimal_policy[i, l, l_b]] * stationary_dist[i, l, l_b]

print(optimal_avg_reward)

total_time = end_time - start_time
print(f'Iteration took {total_time} seconds')

# plot_optimal_policy(optimal_policy, I, L)

# for l in range(L + 1):
#     for l_b in range(L + 1):
#         if l >= l_b:
#             continue
        
#         plt.figure(figsize=(10, 6))

#         i_values = np.arange(2 * I + 1)
#         action_0_values = V_actions[:, l, l_b, 0]
#         action_1_values = V_actions[:, l, l_b, 1]

#         plt.plot(i_values, action_0_values, label='Action 0', marker='o')
#         plt.plot(i_values, action_1_values, label='Action 1', marker='s')

#         plt.xlabel('Inventory Level (i)')
#         plt.ylabel('Value of Actions')
#         plt.title(f'V_actions[i, l={l}, l_b={l_b}, a] vs Inventory Level')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.7)

#         plt.tight_layout()
#         plt.show()
        
# for fixed_i in range(2 * I + 1):
#     for fixed_l in range(L + 1):
#         l_b_values = np.arange(fixed_l + 1, L + 1)
#         if len(l_b_values) == 0:
#             continue  # Skip invalid combinations

#         plt.figure(figsize=(10, 6))
#         plt.plot(l_b_values, V_actions[fixed_i, fixed_l, fixed_l + 1:, 0],
#                  label='Action 0', marker='o')
#         plt.plot(l_b_values, V_actions[fixed_i, fixed_l, fixed_l + 1:, 1],
#                  label='Action 1', marker='s')

#         plt.xlabel('Shelf Life of Incoming Batch (l_b)')
#         plt.ylabel('Value of Actions')
#         plt.title(f'V_actions[i={fixed_i}, l={fixed_l}, l_b, a] vs Incoming Shelf Life (l_b)')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(l_b_values)
#         plt.show()

# Version 2: Fixed i and l_b, x-axis is l
# for fixed_i in range(2 * I + 1):
#     for fixed_l_b in range(1, L + 1):  # start from 1 since l_b must be at least 1
#         l_values = np.arange(0, fixed_l_b)

#         if len(l_values) == 0:
#             continue  # Skip invalid combinations

#         plt.figure(figsize=(10, 6))
#         plt.plot(l_values, V_actions[fixed_i, :fixed_l_b, fixed_l_b, 0],
#                  label='Action 0', marker='o')
#         plt.plot(l_values, V_actions[fixed_i, :fixed_l_b, fixed_l_b, 1],
#                  label='Action 1', marker='s')

#         plt.xlabel('Shelf Life of Current Inventory (l)')
#         plt.ylabel('Value of Actions')
#         plt.title(f'V_actions[i={fixed_i}, l, l_b={fixed_l_b}, a] vs Current Shelf Life (l)')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(l_values)
#         plt.show()




                
                
                

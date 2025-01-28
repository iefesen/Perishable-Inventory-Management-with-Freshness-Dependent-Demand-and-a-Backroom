# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 17:26:45 2024

@author: 20224695+++
+
+++
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Parameters
K = 35
I = 20
L = 8
p = 6
s = 0.4
lambda_c = 12
P_b = 0.3


num_actions = 2

np.random.seed(42)

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



# for i in range(I + 1):
#     for l in range(L + 1):
#         for l_b in range(L + 1):
#             if l >= l_b and l != 0:
#                 continue
#             for a in range(num_actions):    
#                 print(f'Reward of action {a} in state ({i}, {l}, {l_b}) is: {R[i, l, l_b, a]}')

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
                    
# for i in range(I+1):
#     for l in range(L+1):
#         for l_b in range(L+1):
#             if l >= l_b and l != 0:
#                 continue
#             else:
#                 for a in range(num_actions):
#                     sum_prob = 0
#                     for i_ in range(I+1):
#                         for l_ in range(L+1):
#                             for l_b_ in range(L+1):
#                                 sum_prob = sum_prob + P[i, l, l_b, a, i_, l_, l_b_]
#                     if abs(sum_prob - 1) > 1e-6:
#                         print(f'Problem with state {i}, {l}, {l_b} action {a} with prob {sum_prob}')


                        
# for i in range(I+1):
#     for l in range(L+1):
#         for l_b in range(L+1):
#             if l >= l_b and l != 0:
#                 continue
#             else:
#                 for a in range(num_actions):
#                     for i_ in range(I+1):
#                         for l_ in range(L+1):
#                             for l_b_ in range(L+1):
#                                 if P[i, l, l_b, a, i_, l_, l_b_] > 0:
#                                     print(f'State {i}, {l}, {l_b} goes to {i_}, {l_}, {l_b_} with action {a} is {P[i, l, l_b, a, i_, l_, l_b_]}') 
                                                
                    
                    


# for i in range(I+1):
#     for l in range(L+1):
#         for l_b in range(L+1):
#             if l >= l_b and l != 0:
#                 continue
#             for a in range(num_actions):
#                 sum_prob = 0
#                 for i_ in range(I+1):
#                     for l_ in range(L+1):
#                         for l_b_ in range(L+1):
#                             if P[i, l, l_b, a, i_, l_, l_b_]!= 0:
#                                 # print(f'Going from state ({i}, {l}, {l_b}) by action {a} to ({i_}, {l_}, {l_b_}) has a probability of: {P[i, l, l_b, a, i_, l_, l_b_]}')
#                                 # print(P[i, l, l_b, a, i_, l_, l_b_])
#                                 sum_prob = sum_prob + P[i, l, l_b, a, i_, l_, l_b_]
#                 if abs(sum_prob - 1) >1e-6:
#                     print(f'State ({i}, {l}, {l_b}) has {abs(1 - sum_prob)} shortage for action {a}')
                    
                    
                    
                    
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
        print('The value iteration converged at iteration', iteration)
        break
    
'Finding the optimal policy'
optimal_policy = np.argmax(V_actions, axis = 3)




#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

# 'Relative Value Iteration'

# V_rel_actions = np.zeros((I+1, L+1, L+1, num_actions))
# V_rel = np.zeros((I+1, L+1, L+1))

# V_rel_prev = np.copy(V_rel)
# V_rel_actions_prev = np.copy(V_rel_actions)
# iteration = 0


# while True:
#     V_rel_prev = np.copy(V_rel)
#     iteration = iteration + 1
#     min_ = float('inf')
#     for i in range(I+1):
#         for l in range(L+1):
#             for l_b in range(L+1):
#                 if l >= l_b:
#                     continue
#                 else:
#                     if l == 0 or  i == 0 :
#                         V_rel_actions[i, l, l_b, 0] = -float('inf')
#                         V_rel_actions[i ,l, l_b, 1] = R[i, l, l_b, 1] + np.sum(P[i, l, l_b, 1] * V_rel_prev)
#                     else:
#                         for a in range(num_actions):
#                             V_rel_actions[i ,l, l_b, a] = R[i, l, l_b, a] + np.sum(P[i, l, l_b, a] * V_rel_prev)
#     V_rel = np.max(V_rel_actions, axis = 3)
#     max_ = np.max(V_rel - V_rel_prev)
#     for i in range(I+1):
#         for l in range(L+1):
#             for l_b in range(L+1):
#                 if l_b > l and abs(V_rel[i, l, l_b] - V_rel_prev[i, l, l_b]) < min_:
#                     min_ = np.min(V_rel[i, l, l_b] - V_rel_prev[i, l, l_b])
                    
#     if max_ - min_ < 1e-6 or iteration  == 10000:
#         print(f'Average is {(max_ + min_)/2}')
#         print(f'Maximum is {max_}')
#         print(f'Minimum is {min_}')
#         print(f'Iteration is {iteration}')
#         break



    
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


# 'Plot for l_b'        
# for i in range(I+1):
#     for l in range(L+1):
#         for a in range(num_actions):
#             x_values = range(l + 1, V_actions.shape[2])
#             plt.plot(x_values, V_actions[i, l, l+1:, a], label=f'a={a}')
        
#         plt.xlabel('Backroom lifetime')
#         plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
#         plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
#         plt.legend()
#         plt.show()
        

# 'Plot for l'        
# for i in range(I+1):
#     for l_b in range(L+1):
#         for a in range(num_actions):
#             plt.plot(V_actions[i, :l_b, l_b, a], label=f'a={a}')
        
#         plt.xlabel('Shelf lifetime')
#         plt.ylabel(f'V_actions[{i}, l, {l_b}, a]')
#         plt.title(f'The value function in shelf lifetime for each action when i = {i} and l_b = {l_b}')
#         plt.legend()
#         plt.show()

# 'Plot for i'        
# for l in range(L+1):
#     for l_b in range(L+1):
#         if l >= l_b and l != 0:
#             continue
#         else:
#             for a in range(num_actions):
#                 plt.plot(V_actions[:, l, l_b, a], label=f'a={a}')
            
#             plt.xlabel('Inventory Level')
#             plt.ylabel(f'V_actions[i, {l}, {l_b}, a]')
#             plt.title(f'The value function in inventory level for each action when l = {l} and l_b = {l_b}')
#             plt.legend()
#             plt.show()

# for l_b in range(L+1):
#     'Plotting the optimal policy'
#     i_axis = []
#     l_axis = []
    
#     for i in range(I+1):
#         for l in range(L+1):
#             if optimal_policy[i,l, l_b] == 1:
#                 i_axis.append(i)
#                 l_axis.append(l)
                
               
#     plt.scatter(i_axis, l_axis)
#     plt.xlabel('Inventory Level')
#     plt.ylabel('Shelf Lifetime')
#     plt.title(f'Optimal policy for l_b = {l_b}')
    
    
#     x_ticks = []
#     for i in range(I+1):
#         x_ticks.append(i)
        
#     y_ticks = []
#     for l in range(L+1):
#         y_ticks.append(l)
    
#     plt.yticks(y_ticks)
#     plt.xticks(x_ticks)
    
#     plt.xlim(0, I+1)
#     plt.ylim(0, L+1) 
#     plt.show() 


# def simulate_mdp(policy, P, R, start_state, num_steps):
#     state = start_state
#     rewards_obtained = 0
#     per_time_rewards_obtained = float('inf')
#     for step in range(num_steps):
#         i, l, l_b = state
#         action = policy[i, l, l_b]
        
#         # Get the reward for the current state-action pair
#         reward = R[i, l, l_b, action]
#         rewards_obtained = rewards_obtained + reward
        
#         # Get the transition probabilities for the current state-action pair
#         transition_probs = P[i, l, l_b, action]
        
#         # Flatten the transition probabilities to select the next state
#         transition_probs_flat = transition_probs.flatten()
#         next_state_index = np.random.choice(np.arange(transition_probs_flat.size), p=transition_probs_flat)
        
#         # Convert flat index back to state indices
#         next_state = np.unravel_index(next_state_index, (I+1, L+1, L+1))
        
#         # Update state
#         state = next_state
#         per_time_rewards_obtained_new = rewards_obtained / (step + 1)
#         if abs(per_time_rewards_obtained - per_time_rewards_obtained_new) <1e-4 and step > 1000:
#             per_time_rewards_obtained = per_time_rewards_obtained_new
#             break
#         else:
#             per_time_rewards_obtained = per_time_rewards_obtained_new
#     return per_time_rewards_obtained

# # Parameters
# max_simulations = 1000  # Maximum number of simulations
# num_steps = 100000  # Maximum number of steps per simulation
# start_state = (0, 0, L)  # Starting state (you can change this)

# # Run the simulation
# total_of_sims = 0
# avg_of_sims = float('inf')

# for sim in range(max_simulations):
#     per_time_rewards_obtained = simulate_mdp(optimal_policy, P, R, start_state, num_steps)
#     total_of_sims = total_of_sims + per_time_rewards_obtained
#     avg_of_sims_new = total_of_sims / (sim + 1)
#     if abs(avg_of_sims - avg_of_sims_new) < 1e-4:
#         avg_of_sims = avg_of_sims_new
#         print('Average reward is: ', avg_of_sims)
#         print('Converged at simulation: ', sim + 1)
#         break
#     else:
#         # print('The difference is: ', abs(avg_of_sims - avg_of_sims_new))
#         avg_of_sims = avg_of_sims_new
        
        
# for i_t in range(I+1):
#     # Run the simulation
#     total_of_sims = 0
#     avg_of_sims = float('inf')
#     i_t_threshold = np.zeros((I+1, L+1, L+1))
#     for _ in range(i_t + 1):
#         i_t_threshold[_, :, :] = 1
#     i_t_threshold[:, 0, :] = 1
#     for sim in range(max_simulations):
#         per_time_rewards_obtained = simulate_mdp(optimal_policy, P, R, start_state, num_steps)
#         total_of_sims = total_of_sims + per_time_rewards_obtained
#         avg_of_sims_new = total_of_sims / (sim + 1)
#         if abs(avg_of_sims - avg_of_sims_new) < 1e-4:
#             avg_of_sims = avg_of_sims_new
#             print('Average reward is: ', avg_of_sims)
#             print('Converged at simulation: ', sim + 1)
#             break
#         else:
#             # print('The difference is: ', abs(avg_of_sims - avg_of_sims_new))
#             avg_of_sims = avg_of_sims_new
# 

# max_steps = 10000
# max_sims = 10
# arrivals = np.zeros((max_steps, max_sims))
# decay_backroom = np.zeros((max_steps, max_sims))

# for step in range(max_steps):
#     for sim in range(max_sims):
#         arrivals[step, sim] = np.random.poisson(lambda_c)
#         decay_backroom[step, sim] = np.random.choice([0, 1], p=[1 - P_b, P_b])
        
        
# def simulate(policy, P, R, max_steps, max_sims, start_state):
#     total_of_sims = 0
#     avg_of_sims = float('inf')
    
#     for sim in range(max_sims):
#         state = start_state[:]  # Reset the state at the start of each simulation
#         sim_reward = 0
#         sim_reward_avg = float('inf')
        
#         for step in range(max_steps):
#             i = int(state[0])
#             l = int(state[1])
#             l_b = int(state[2])
#             action = policy[i, l, l_b]
            
#             if action == 1:
#                 sim_reward += -K + s*i
#                 i = I
#                 l = l_b
#                 l_b = L
            
#             attract = arrivals[step, sim] * (l/L)
#             if attract >= i:
#                 sim_reward += i * p
#                 i = 0
#             else:
#                 sim_reward += attract * p
#                 i -= attract
            
#             if l != 0:
#                 l -= 1
#             if l_b != 0:
#                 l_b -= decay_backroom[step, sim]
            
#             state[0] = i
#             state[1] = l
#             state[2] = l_b
            
#             sim_reward_avg_new = sim_reward / (step + 1)
#             if abs(sim_reward_avg - sim_reward_avg_new) < 1e-4:
#                 break
#             sim_reward_avg = sim_reward_avg_new
        
#         total_of_sims += sim_reward_avg
        
#         avg_of_sims_new = total_of_sims / (sim + 1)
#         if abs(avg_of_sims - avg_of_sims_new) < 1e-4:
#             avg_of_sims = avg_of_sims_new
#             break
#         avg_of_sims = avg_of_sims_new
    
#     return avg_of_sims


# policy_1_threshold = np.zeros((I+1, L+1, L+1))
# policy_1_threshold[:, 0, :] = 1
# policy_1_threshold[0, :, :] = 1
# policy_1_threshold[1, :, :] = 1



# simulate(optimal_policy, P, R, max_steps, max_sims, [0, L , L] )            
# simulate(policy_1_threshold, P, R, max_steps, max_sims, [0, L , L] )       


    
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################
# Simulation parameters
num_steps = 10000  # Number of steps in each simulation

# Convergence parameters
epsilon = 1e-3  # Convergence threshold
max_simulations = 1000  # Maximum number of simulations to avoid infinite loops
simulations_run = 0
previous_avg_reward = 0
average_rewards_per_simulation = []

while simulations_run < max_simulations:
    simulations_run += 1
    state = [0, 0, L]
    
    total_reward = 0
    
    for _ in range(num_steps):
        i, l, l_b = state
        action = optimal_policy[i, l, l_b]  # Select the optimal action
        total_reward += R[i, l, l_b, action]
        
        # Determine the next state
        next_state_probs = P[i, l, l_b, action]
        flat_probs = next_state_probs.flatten()

        # Normalize the probabilities to ensure they sum to 1
        flat_probs_sum = flat_probs.sum()

        # Handle potential NaN or zero-probability issues
        if flat_probs_sum == 0 or np.isnan(flat_probs_sum):
            print("Warning: Transition probabilities are zero or contain NaN.")
            break

        flat_probs /= flat_probs_sum

        if np.any(np.isnan(flat_probs)):
            print("Warning: NaN found in transition probabilities after normalization.")
            break

        next_state_index = np.random.choice(range(flat_probs.size), p=flat_probs)
        i_, l_, l_b_ = np.unravel_index(next_state_index, (I+1, L+1, L+1))
        state = (i_, l_, l_b_)
    
    # Calculate the average reward per step for this simulation
    avg_reward_per_step = total_reward / num_steps
    average_rewards_per_simulation.append(avg_reward_per_step)
    
    # Calculate the overall average reward per step across all simulations
    current_avg_reward = np.mean(average_rewards_per_simulation)
    
    # Check for convergence
    if abs(current_avg_reward - previous_avg_reward) < epsilon:
        # print(f"Converged after {simulations_run} simulations")
        break
    
    previous_avg_reward = current_avg_reward

# Output the final average reward per step
print(f"Optimal average reward per step: {current_avg_reward:.6f}")
    
#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################    
# Simulation parameters
num_steps = 10000  # Number of steps in each simulation

# Convergence parameters
epsilon = 1e-3  # Convergence threshold
max_simulations = 1000  # Maximum number of simulations to avoid infinite loops
simulations_run = 0
previous_avg_reward = 0
average_rewards_per_simulation = []

for i_t in range(I+1):
    while simulations_run < max_simulations:
        simulations_run += 1
        state = [0, 0, L]
        
        total_reward = 0
        
        for _ in range(num_steps):
            i, l, l_b = state
            if i <= i_t or l == 0:
                action = 1
            else:
                action = 0
            # Select the optimal action
            total_reward += R[i, l, l_b, action]
            
            # Determine the next state
            next_state_probs = P[i, l, l_b, action]
            flat_probs = next_state_probs.flatten()
    
            # Normalize the probabilities to ensure they sum to 1
            flat_probs_sum = flat_probs.sum()
    
            # Handle potential NaN or zero-probability issues
            if flat_probs_sum == 0 or np.isnan(flat_probs_sum):
                print("Warning: Transition probabilities are zero or contain NaN.")
                break
    
            flat_probs /= flat_probs_sum
    
            if np.any(np.isnan(flat_probs)):
                print("Warning: NaN found in transition probabilities after normalization.")
                break
    
            next_state_index = np.random.choice(range(flat_probs.size), p=flat_probs)
            i_, l_, l_b_ = np.unravel_index(next_state_index, (I+1, L+1, L+1))
            state = (i_, l_, l_b_)
        
        # Calculate the average reward per step for this simulation
        avg_reward_per_step = total_reward / num_steps
        average_rewards_per_simulation.append(avg_reward_per_step)
        
        # Calculate the overall average reward per step across all simulations
        current_avg_reward = np.mean(average_rewards_per_simulation)
        
        # Check for convergence
        if abs(current_avg_reward - previous_avg_reward) < epsilon:
            # print(f"Converged after {simulations_run} simulations")
            break
        
        previous_avg_reward = current_avg_reward
    
    # Output the final average reward per step
    print(f"Final average reward per step for threshold {i_t}: {current_avg_reward:.6f}") 


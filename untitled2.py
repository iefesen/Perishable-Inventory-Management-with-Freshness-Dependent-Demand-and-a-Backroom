import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib.patches as mpatches


# Define parameters
K = 250
I = 30
L = 5
p = 15
s = 4
lambda_c = 6
P_b = 0.1


num_actions = 2

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

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################

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



#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


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
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################


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
        break
    
optimal_policy = np.argmax(V_actions, axis = 3)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################



'Plot for l_b'        
for i in range(I+1):
    for l in range(L+1):
        for a in range(num_actions):
            x_values = range(l + 1, V_actions.shape[2])
            plt.plot(x_values, V_actions[i, l, l+1:, a], label=f'a={a}')
        
        plt.xlabel('Backroom lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
        plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
        plt.legend()
        plt.show()
        

'Plot for l'        
for i in range(I+1):
    for l_b in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[i, 1:l_b, l_b, a], label=f'a={a}')
        
        plt.xlabel('Shelf lifetime')
        plt.ylabel(f'V_actions[{i}, l, {l_b}, a]')
        plt.title(f'The value function in shelf lifetime for each action when i = {i} and l_b = {l_b}')
        plt.legend()
        plt.show()

'Plot for i'        
for l in range(L+1):
    for l_b in range(L+1):
        if l >= l_b and l != 0:
            continue
        else:
            for a in range(num_actions):
                plt.plot(V_actions[:, l, l_b, a], label=f'a={a}')
            
            plt.xlabel('Inventory Level')
            plt.ylabel(f'V_actions[i, {l}, {l_b}, a]')
            plt.title(f'The value function in inventory level for each action when l = {l} and l_b = {l_b}')
            plt.legend()
            plt.show()

'Plot for l_b'        
for i in range(I+1):
    for l in range(L+1):
        for a in range(num_actions):
            x_values = range(l + 1, V_actions.shape[2])
            plt.plot(x_values, V_actions[i, l, l+1:, int(optimal_policy[i, l, l_b])], label=f'a={a}')
        
        plt.xlabel('Backroom lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
        plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
        plt.legend()
        plt.show()
        

'Plot for l'        
for i in range(I+1):
    for l_b in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[i, 1:l_b, l_b, int(optimal_policy[i, l, l_b])], label=f'a={a}')
        
        plt.xlabel('Shelf lifetime')
        plt.ylabel(f'V_actions[{i}, l, {l_b}, a]')
        plt.title(f'The value function in shelf lifetime for each action when i = {i} and l_b = {l_b}')
        plt.legend()
        plt.show()

'Plot for i'        
for l in range(L+1):
    for l_b in range(L+1):
        if l >= l_b and l != 0:
            continue
        else:
            for a in range(num_actions):
                plt.plot(V_actions[:, l, l_b, int(optimal_policy[i, l, l_b])], label=f'a={a}')
            
            plt.xlabel('Inventory Level')
            plt.ylabel(f'V_actions[i, {l}, {l_b}, a]')
            plt.title(f'The value function in inventory level for each action when l = {l} and l_b = {l_b}')
            plt.legend()
            plt.show()        

# for i in range(I+1):
#     for l in range(L+1):
#         # Define the x-axis range based on the backroom lifetime (l_b)
#         l_b_axis = list(range(l+1, L+1))  # x-axis values corresponding to the backroom lifetime

#         # Plotting for action 0 and 1 with blue and red lines respectively
#         plt.plot(l_b_axis, V_actions[i, l, (l+1):, 0], label='Continue', color='blue')  # Action 0 in blue
#         plt.plot(l_b_axis, V_actions[i, l, (l+1):, 1], label='Replenish', color='red')  # Action 1 in red
        
#         # Labels for axes
#         plt.xlabel('Backroom Lifetime')
#         plt.ylabel('Total Discounted Rewards')

#         # Customizing legend
#         plt.legend()

#         # Set x-axis ticks to reflect actual values of l_b
#         plt.xticks(l_b_axis)  # Ensure ticks are at actual l_b values

#         # Remove the top and right spines of the plot box (keep left and bottom spines)
#         ax = plt.gca()  # Get the current axis
#         ax.spines['top'].set_visible(False)  # Hide top spine
#         ax.spines['right'].set_visible(False)  # Hide right spine

#         # Save the figure with high quality (300 DPI)
#         plt.savefig(f'discounted_rewards_i_{i}_l_{l}.png', dpi=300, bbox_inches='tight')

#         # Show the plot
#         plt.show()

# for l_b in range(L+1):
#     # Lists to store the maximum points for each i
#     i_axis = []
#     max_l_axis = []
    
#     for i in range(I+1):
#         # Find the maximum l where the optimal policy is 1 for each inventory level i
#         l_values = [l for l in range(L+1) if optimal_policy[i, l, l_b] == 1]
#         if l_values:  # Check if there are any valid l for this i
#             i_axis.append(i)
#             max_l_axis.append(max(l_values))  # Store the maximum l for this i
    
#     # Plot the line in black and label it as "Continue Region"
#     plt.plot(i_axis, max_l_axis, linestyle='-', color='black', label='Continue Region')
    
#     # Fill the area below the line with black and diagonal hatching
#     plt.fill_between(i_axis, max_l_axis, color='white', edgecolor='black', alpha=0.3, hatch='///', label='Replenishment Region')
    
#     plt.xlabel('Inventory Level')
#     plt.ylabel('Shelf Lifetime')
    
#     # Setting custom x-ticks for intervals of 5
#     x_ticks = list(range(0, I+1, 5))
#     y_ticks = list(range(L+1))
    
#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks)
    
#     plt.xlim(0, I+1)
#     plt.ylim(0, L+1) 
    
#     # Remove the top and right spines
#     ax = plt.gca()  # Get current axis
#     ax.spines['top'].set_visible(False)  # Hide top spine
#     ax.spines['right'].set_visible(False)  # Hide right spine

#     # Create custom handles for the legend
#     line_handle = mpatches.Patch(facecolor='white', edgecolor='black', label='Continue Region')  # White box for continue region
#     hatch_handle = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Replenishment Region')  # Hatched box matching the plot for replenishment region

#     # Add the legend with smaller font size
#     plt.legend(handles=[line_handle, hatch_handle], loc='upper right', fontsize='small')

#     # Save the figure with high quality (300 DPI)
#     plt.savefig(f'optimal_policy_lb_{l_b}.png', dpi=300, bbox_inches='tight')
    
#     plt.show()




    
# def calculate_stationary_distribution(P, policy, I, L):
#     num_states = (I + 1) * (L + 1) * (L + 1)
#     P_flat = np.zeros((num_states, num_states))
    
#     for i in range(I + 1):
#         for l in range(L + 1):
#             for l_b in range(L + 1):
#                 if l >= l_b:
#                     continue
#                 state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                 action = policy[i, l, l_b]
                
#                 for i_next in range(I + 1):
#                     for l_next in range(L + 1):
#                         for l_b_next in range(L + 1):
#                             next_state_index = i_next * (L + 1) * (L + 1) + l_next * (L + 1) + l_b_next
#                             P_flat[state_index, next_state_index] = P[i, l, l_b, action, i_next, l_next, l_b_next]
    
#     row_sums = P_flat.sum(axis=1)
    
#     zero_sum_rows = (row_sums == 0)
#     P_flat[zero_sum_rows, :] = 1 / num_states
#     row_sums[zero_sum_rows] = 1
    
#     P_flat = P_flat / row_sums[:, np.newaxis]
    
#     if np.any(np.isnan(P_flat)) or np.any(np.isinf(P_flat)):
#         P_flat = np.nan_to_num(P_flat, nan=0, posinf=0, neginf=0)
#         row_sums = P_flat.sum(axis=1)
#         P_flat = P_flat / row_sums[:, np.newaxis]
    
#     eigenvalues, eigenvectors = linalg.eig(P_flat.T)
#     index = np.argmin(np.abs(eigenvalues - 1))
#     stationary_dist = eigenvectors[:, index].real
#     stationary_dist = stationary_dist / np.sum(stationary_dist)
    
#     stationary_dist_shaped = np.zeros((I + 1, L + 1, L + 1))
#     for i in range(I + 1):
#         for l in range(L + 1):
#             for l_b in range(L + 1):
#                 if l >= l_b:
#                     continue
#                 state_index = i * (L + 1) * (L + 1) + l * (L + 1) + l_b
#                 stationary_dist_shaped[i, l, l_b] = stationary_dist[state_index]
    
#     return stationary_dist_shaped

# # Calculate optimal average reward

# stationary_dist = calculate_stationary_distribution(P, optimal_policy, I, L)

# optimal_avg_reward = 0





# for l in range(L+1):
#     'Plotting the optimal policy'
#     max_i_per_l_b = {}

#     for l_b in range(L+1):
#         max_i = None
#         for i in range(I+1):
#             if optimal_policy[i, l, l_b] == 1:
#                 max_i = i  # Update max_i whenever optimal_policy is 1
#         if max_i is not None:
#             max_i_per_l_b[l_b] = max_i  # Store the maximum i for each l_b

#     # Extract the lists of i and l_b for plotting
#     l_b_axis = list(max_i_per_l_b.keys())
#     i_axis = [max_i_per_l_b[l_b] for l_b in l_b_axis]

#     # Plot the line connecting the points
#     plt.plot(i_axis, l_b_axis, '-r', label='Max Line', color='black')  # Black line

#     # Add hatching to the left of the line
#     plt.fill_betweenx(l_b_axis, 0, i_axis, hatch='///', facecolor='white', edgecolor='black', alpha=0.3)

#     plt.xlabel('Inventory Level')
#     plt.ylabel('Backroom Lifetime')

#     # Custom x-ticks for intervals of 5 and y-ticks for every l_b
#     x_ticks = list(range(0, I+1, 5))
#     y_ticks = list(range(L + 1))

#     plt.xticks(x_ticks)
#     plt.yticks(y_ticks)

#     plt.xlim(0, I + 1)
#     plt.ylim(l + 1, L + 1)

#     # Remove the top and right spines
#     ax = plt.gca()  # Get current axis
#     ax.spines['top'].set_visible(False)  # Hide top spine
#     ax.spines['right'].set_visible(False)  # Hide right spine

#     # Create custom handles for the legend
#     line_handle = mpatches.Patch(facecolor='white', edgecolor='black', label='Continue Region')
#     hatch_handle = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='Replenishment Region')

#     # Add the legend with smaller font size
#     plt.legend(handles=[line_handle, hatch_handle], loc='upper right', fontsize='small')

#     # Save the figure with high quality (300 DPI)
#     plt.savefig(f'optimal_policy_l_{l}.png', dpi=300, bbox_inches='tight')

#     plt.show()

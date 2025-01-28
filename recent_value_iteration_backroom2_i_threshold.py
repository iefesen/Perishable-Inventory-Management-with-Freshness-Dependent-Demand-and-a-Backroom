import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


# Parameters
K = 35
I = 10
L = 10
p = 4
s = 1
lambda_c = 10
P_b = 0.99
num_actions = 2

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

'Value iteration parameters'
epsilon = 1e-10
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
                for a in range(num_actions):
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

# Create the transition matrix for the optimal policy
P_opt = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))

for i in range(I + 1):
    for l in range(L + 1):
        for l_b in range(L + 1):
            P_opt[i, l, l_b] = P[i, l, l_b, optimal_policy[i, l, l_b]]

P_opt_flat = P_opt.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))
steady_state_opt = compute_steady_state_distribution(P_opt_flat)
# For the optimal policy
P_opt = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))

for i in range(I + 1):
    for l in range(L + 1):
        for l_b in range(L + 1):
            P_opt[i, l, l_b] = P[i, l, l_b, optimal_policy[i, l, l_b]]

P_opt_flat = P_opt.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))

steady_state_opt = compute_steady_state_distribution(P_opt_flat)
steady_state_opt = steady_state_opt.reshape((I + 1, L + 1, L + 1))

opt_average = 0

for i in range(I + 1):
    for l in range(L + 1):
        for l_b in range(L + 1):
            opt_average += R[i, l, l_b, optimal_policy[i, l, l_b]] * steady_state_opt[i, l, l_b]

print("Average Reward Optimal:", opt_average)


#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


for i_t in range(I + 1):
    P_threshold = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))
    for i in range(I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                if i < i_t or l == 0:
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
                if i < i_t or l == 0:
                    average += R[i, l, l_b, 1] * steady_state[i, l, l_b]
                else:
                    average += R[i, l, l_b, 0] * steady_state[i, l, l_b]

    print(f'Average Reward Threshold {i_t}: {average}')

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################


'Plot for l_b'        
for i in range(I+1):
    for l in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[i, l, l:, a], label=f'a={a}')
        
        plt.xlabel('Backroom lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
        plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
        plt.legend()
        plt.gca().invert_xaxis()  # Invert the x-axis
        plt.show()
        
print(V_actions[9, 2, 6, 1])
print(V_actions[9, 2, 6, 0])

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



    
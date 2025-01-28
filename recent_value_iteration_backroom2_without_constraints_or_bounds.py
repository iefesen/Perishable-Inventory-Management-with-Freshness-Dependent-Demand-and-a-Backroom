
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt


# Parameters
K = 25
I = 20
L = 10
p = 5
s = 1
lambda_c = 5
P_b = 1
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
                if a == 1:
                    R[i, l, l_b, a] = - K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
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
                if a == 1 and l_b == 0:
                    P[i, l, l_b, a, i, l_b, L - 1] =  P_b
                    P[i, l, l_b, a, i, l_b, L] =  1 - P_b
                else:
                    if l == 0 or i == 0:
                        if l_b != 0:
                            P[i, l, l_b, a, i, l, l_b] = 1 - P_b
                            P[i, l, l_b, a, i, l, l_b - 1] = P_b
                        else:
                            P[i, l, l_b, a, i, l, l_b] = 1
                    else:
                        P[i, l, l_b, a, 0, l - 1, l_b - 1] = arr_greater_than(l, i - 1) * P_b
                        for k in range(i):
                            P[i, l, l_b, a, i - k, l - 1, l_b - 1] = arrival_probability(l, k) * P_b
                        P[i, l, l_b, a, 0, l - 1, l_b] = arr_greater_than(l, i - 1) * (1 - P_b)
                        for k in range(i):
                            P[i, l, l_b, a, i - k, l - 1, l_b] = arrival_probability(l, k) * (1 - P_b)

                        

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
#         print('The value iteration converged at iteration', iteration)
#         break
    
# 'Finding the optimal policy'
# optimal_policy = np.argmax(V_actions, axis = 3)
# Value iteration
V = np.zeros((I + 1, L + 1, L + 1))
V_actions = np.zeros((I + 1, L + 1, L + 1, num_actions))

def value_iteration(V, V_actions, R, P, gamma, theta):
    delta = float('inf')
    while delta > theta:
        delta = 0
        for i in range(I + 1):
            for l in range(L + 1):
                for l_b in range(L + 1):
                    if l > l_b:
                        continue
                    v = V[i, l, l_b]
                    for a in range(num_actions):
                        V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V)
                    V[i, l, l_b] = np.max(V_actions[i, l, l_b])
                    delta = max(delta, abs(v - V[i, l, l_b]))
    return V, V_actions

V, V_actions = value_iteration(V, V_actions, R, P, gamma = 0.99, theta = 1e-10)

optimal_policy = np.argmax(V_actions, axis = 3)

#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

'Plot for l_b'        
for i in range(I+1):
    for l in range(L+1):
        for a in range(num_actions):
            x_values = range(l, l + V_actions.shape[2] - l)
            plt.plot(x_values, V_actions[i, l, l:, a], label=f'a={a}')
        
        plt.xlabel('Backroom lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
        plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
        plt.legend()
        plt.show()
        
        
# 'Plot for l'        
# for i in range(I+1):
#     for l_b in range(L+1):
#         for a in range(num_actions):
#             plt.plot(V_actions[i, :l_b+1, l_b, a], label=f'a={a}')
        
#         plt.xlabel('Shelf lifetime')
#         plt.ylabel(f'V_actions[{i}, l, {l_b}, a]')
#         plt.title(f'The value function in backroom lifetime for each action when i = {i} and l_b = {l_b}')
#         plt.legend()
#         plt.show()


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




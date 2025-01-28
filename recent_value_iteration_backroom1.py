# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:06:11 2024

@author: 20224695
"""


import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

K = 10

'Setting the parameters for the problem environment'
I = 10
L = 15
p = 3
s = 1
lambda_c = 1
lambda_s = 2

num_actions = 2

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



#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

R = np.zeros((I + 1, L + 1, L+1, num_actions))
for i in range(I + 1):
    for l in range(L + 1):
        for l_b in range(L+1):
            if l_b < l:
                continue
            for a in range(num_actions):
                if a == 1:
                    R[i, l, l_b, a] = -K + s * i + p * (sum(k * arrival_probability(l_b, k) for k in range(I))) + arr_greater_than(l_b, I - 1) * I * p
                else:
                    R[i, l, l_b, a] = p * (sum(k * arrival_probability(l, k) for k in range(i))) + arr_greater_than(l, i - 1) * i * p

            
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------- 


# Empty set containing the transition probabilities for each state
P = np.zeros((I + 1, L + 1, L + 1, num_actions, I + 1, L + 1, L + 1))

# Calculating the transition probabilities
for i in range(I + 1):
    for l in range(L + 1):
        for l_b in range(L+1):
            if l_b < l:
                continue
            for a in range(num_actions):
                if a == 1:
                    if l_b >= lambda_s:
                        for k in range(I):
                            P[i, l, l_b, a, I - k, l_b - lambda_s, L - 1] = arrival_probability(l_b, k)
                        P[i, l, l_b, a, 0, l_b - lambda_s, L - 1] = arr_greater_than(l_b, I - 1)
                    else:
                        for k in range(I):
                            P[i, l, l_b, a, I - k, 0, L - 1] = arrival_probability(l_b, k)
                        P[i, l, l_b, a, 0, 0, L - 1] = arr_greater_than(l_b, I - 1)
                else:
                    if l == 0:
                        P[i, l, l_b, a, i, 0, l_b - 1] = 1
                    else:
                        if l >= lambda_s:
                            for k in range(i):
                                P[i, l, l_b, a, i - k, l - lambda_s, l_b - 1] = arrival_probability(l, k)
                            P[i, l, l_b, a, 0, l - lambda_s, l_b - 1] = arr_greater_than(l, i - 1)
                        else:
                            for k in range(i):
                                P[i, l, l_b, a, i - k, 0, l_b - 1] = arrival_probability(l, k)
                            P[i, l, l_b, a, 0, 0, l_b - 1] = arr_greater_than(l, i - 1)
                            

# for i in range(I+1):
#     for l in range(L+1):
#         for l_b in range(L+1):
#             for i_ in range(I+1):
#                 for l_ in range(L+1):
#                     for l_b_ in range(L+1):
#                         if P[i, l, l_b, 0, i_, l_,l_b_] != 0:
#                             print(f'Probability of moving from ({i}, {l}, {l_b}) to ({i_}, {l_}, {l_b}) with action 0 is {P[i, l, l_b, 0, i_, l_,l_b_]}')


           
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------

# 'Value iteration parameters'
epsilon = 1e-6
gamma = 0.99

'Empty sets containing the value of each state'
V = np.zeros((I+1,L+1, L+1))
V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))

'Value iteration'
while True:
    V_prev = np.copy(V)
    for i in range(I + 1):
        for l in range(L + 1):
            for l_b in range(L + 1):
                for a in range(num_actions):
                    V_actions[i, l, l_b, a] = R[i, l, l_b, a] + gamma * np.sum(P[i, l, l_b, a] * V_prev)
            
    V = np.max(V_actions, axis = 3)
    if np.max(np.abs(V - V_prev)) < epsilon:
        # print('The value iteration converged at iteration', iteration, 'with ', np.max(np.abs(V - V_prev)))
        break


'Finding the optimal policy'
optimal_policy = np.argmax(V_actions, axis = 3)


#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
  
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
    
    
# for l in range(L+1):
#     'Plotting the optimal policy'
#     i_axis = []
#     lb_axis = []
    
#     for i in range(I+1):
#         for l_b in range(l,L+1):
#             if optimal_policy[i,l, l_b] == 1:
#                 i_axis.append(i)
#                 lb_axis.append(l_b)
                
               
#     plt.scatter(i_axis, lb_axis)
#     plt.xlabel('Inventory Level')
#     plt.ylabel('Backroom Lifetime')
#     plt.title(f'Optimal policy for l = {l}')
    
    
#     x_ticks = []
#     for i in range(I+1):
#         x_ticks.append(i)
        
#     y_ticks = []
#     for l_b in range(l, L+1):
#         y_ticks.append(l_b)
    
#     plt.yticks(y_ticks)
#     plt.xticks(x_ticks)
    
#     plt.xlim(0, I+1)
#     plt.ylim(l, L+1) 
#     plt.show() 
    
# for i in range(I+1):
#     'Plotting the optimal policy'
#     l_axis = []
#     lb_axis = []
    
#     for l in range(L+1):
#         for l_b in range(L+1):
#             if optimal_policy[i, l, l_b] == 1:
#                 l_axis.append(l)
#                 lb_axis.append(l_b)
                
               
#     plt.scatter(l_axis, lb_axis)
#     plt.xlabel('Shelf Lifetime')
#     plt.ylabel('Backroom Lifetime')
#     plt.title(f'Optimal policy for i = {i}')
    
    
#     x_ticks = []
#     for l in range(L+1):
#         x_ticks.append(l)
        
#     y_ticks = []
#     for l_b in range(L+1):
#         y_ticks.append(l_b)
    
#     plt.yticks(y_ticks)
#     plt.xticks(x_ticks)
    
#     plt.xlim(0, L+1)
#     plt.ylim(0, L+1) 
#     plt.show() 
    
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------  
         
'Plot for l_b'        
for i in range(I+1):
    for l in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[i, l, :, a], label=f'a={a}')
        
        plt.xlabel('Backroom lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, l_b, a]')
        plt.title(f'The value function in backroom lifetime for each action when i = {i} and l = {l}')
        plt.legend()
        plt.show()
        
'Plot for l'
for i in range(I+1):
    for l_b in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[i, :, l_b, a], label=f'a={a}')
        
        plt.xlabel('Shelf Lifetime')
        plt.ylabel(f'V_actions[{i}, {l}, :, a]')
        plt.title(f'The value function in shelf lifetime for each action when i = {i} and l_b = {l_b}')
        plt.legend()
        plt.show()


'Plot for i'
for l_b in range(L+1):
    for l in range(L+1):
        for a in range(num_actions):
            plt.plot(V_actions[:, l, l_b, a], label=f'a={a}')
        
        plt.xlabel('Inventory Level')
        plt.ylabel(f'V_actions[i, {l}, {l_b}, a]')
        plt.title(f'The value function in backroom lifetime for each action when l_b = {l_b} and l = {l}')
        plt.legend()
        plt.show()
            
        
import numpy as np
from scipy.stats import poisson
import pandas as pd

# Arrival probability functions
def arrival_probability(L_i, k):
    arr = lambda_c * (L_i / L)
    return poisson.pmf(k, arr)

def arr_greater_than(L_i, k):
    arr = lambda_c * (L_i / L)
    return 1 - poisson.cdf(k, arr)

def compute_steady_state_distribution(transition_matrix):
    num_states = (I+1)*(L+1)*(L+1)
    
    # Construct the matrix A for the equation A * pi = b
    A = np.eye(num_states) - transition_matrix.T
    A[-1, :] = 1  # Replace the last row with 1s to ensure a valid probability distribution
    
    b = np.zeros(num_states)
    b[-1] = 1  # The sum of probabilities is 1
    
    try:
        if np.linalg.matrix_rank(A) == num_states:
            steady_state_distribution = np.linalg.solve(A, b)
        else:
            print("Matrix is singular, using least squares solution")
            steady_state_distribution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as e:
        print("Linear algebra error:", e)
        steady_state_distribution = None
    
    return steady_state_distribution

# Initialize parameters
num_actions = 2
case = 0

results = []

# K_range = [10, 20]
# I_range = [10]
# L_range = [5]
# p_range = [4]
# s_range = [0.1]
# lambda_c_range = [5]
# P_b_range = [0.1]

K_range = [100, 200]
I_range = [50, 100]
L_range = [5, 10]
p_range = [3, 10]
s_range = [0.5, 1]
lambda_c_range = [15, 30]
P_b_range = [0.3, 0.7]

for K in K_range:
    for I in I_range:
        for L in L_range:
            for p in p_range:
                for s in s_range:
                    for lambda_c in lambda_c_range:
                        for P_b in P_b_range:
                            case = case + 1
                            if case < 51:
                                continue
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
                            epsilon = 1e-6
                            gamma = 0.99
                            
                            # Empty sets containing the value of each state
                            V = np.zeros((I+1,L+1, L+1))
                            V_actions = np.zeros((I + 1, L + 1, L + 1 , num_actions))
                            
                            # Value iteration
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
                                
                            # Finding the optimal policy
                            optimal_policy = np.argmax(V_actions, axis = 3)
                            
                            # Average reward analysis
                            # Create the transition matrix for the optimal policy
                            P_opt = np.zeros((I + 1, L + 1, L + 1, I + 1, L + 1, L + 1))

                            for i in range(I + 1):
                                for l in range(L + 1):
                                    for l_b in range(L + 1):
                                        P_opt[i, l, l_b] = P[i, l, l_b, optimal_policy[i, l, l_b]]

                            P_opt_flat = P_opt.reshape(((I + 1) * (L + 1) * (L + 1), (I + 1) * (L + 1) * (L + 1)))
                            steady_state_opt = compute_steady_state_distribution(P_opt_flat)
                            
                            if steady_state_opt is not None:
                                steady_state_opt = steady_state_opt.reshape((I + 1, L + 1, L + 1))

                                opt_average = 0

                                for i in range(I + 1):
                                    for l in range(L + 1):
                                        for l_b in range(L + 1):
                                            opt_average += R[i, l, l_b, optimal_policy[i, l, l_b]] * steady_state_opt[i, l, l_b]
                            
                                row = [K, I, L, p, s, lambda_c, P_b, opt_average]
                                results.append(row)
                            else:
                                print(f"Steady state distribution computation failed for case {case}")
                            print(case)

columns = ['K', 'I', 'L', 'p', 's', 'lambda_c', 'P_b', 'opt_average']

results_df = pd.DataFrame(results)
results_df.to_excel('average_reward_leadtime_setting3.xlsx', index=False)

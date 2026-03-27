'''
Code that gives the fit uncertainties in the paper

Details: We get the chosen, d_L, and CMB solutions and calculate, using Monte
Carlo, the range of which the parameter values can range while remaining within
a predefined domain based on percent error
'''

import GA_code_definitions as GA
import random
import time

start_time = time.time()

random.seed(time.time())

# Chosen solution parameters

N = 1.1763439409676324e-05
n_s = 0.9710157846057226
z = 1091.0000000182768
p_b = 0.0420526607268939
s_T= 446.869965087166
A = 0.26017814678535256
b = 0.21395983543637556
z_0 = 12.2
tau = 0.15385543699727866

'''
# CMB solution parameters
N = 1.177488298576863e-05
n_s = 0.9705897016634416
z = 1091.0000002379138
p_b = 0.042055247509368114
s_T= 446.4321303555796
A = 0.2619037711062232
b = 0.21567339570232666
z_0 = 12.2
tau = 0.21196838950216068
'''
'''
# HT solution parameters
N = 1.1731356156226647e-05
n_s = 0.939812985483796
z = 1091.000000230978
p_b = 0.03917986260163099
s_T= 425.61624532699864
A = 0.2685895047030513
b = 0.22358878378167707
z_0 = 12.200000900522857
tau = 0.23574842791179348
'''

# Creates a list for the parameters to be stored in for the given solution

original_list = [N, n_s, z, p_b, s_T, A, b, z_0, tau]

# Calculates the fitnesses for the set of parameters

d_L_fit = GA.d_L_chi2_eval(A, b, z_0, tau)
CMB_fit = GA.CMB_chi2_eval(N, n_s, z, p_b, s_T, A, b, z_0, tau)

# Using Monte Carlo, we get a range of fit uncertainties

num_of_guess_per_par = 1000
num_of_par = 9

# Bounds used in the genetic algorithm

lower_bound = [0.00001, 0.9, 1091.0, .01, 1.0, 0.01, 0.01, 12.19999, 0.1]
upper_bound = [0.00002, 0.99, 1093.0, .1, 600.0, 0.5, 1.0, 12.20001, 1]

# Initializes the list that will hold the uncertainties. Starts off with no
# uncertanty

error_list = [[N, N],[n_s, n_s],[z, z],[p_b, p_b],[s_T, s_T],
              [A, A],[b, b],[z_0, z_0],[tau, tau]]

count = 0 # Number of hits

for i in range(num_of_guess_per_par * 9):
    
    # Determines the parameter being checked
    
    parameter = i // num_of_guess_per_par
    
    # Normalizes the parameters to a scale of 0 to 1
        
    list_par = [(N - lower_bound[0]) / (upper_bound[0] - lower_bound[0]), 
                (n_s - lower_bound[1]) / (upper_bound[1] - lower_bound[1]), 
                (z - lower_bound[2]) / (upper_bound[2] - lower_bound[2]), 
                (p_b - lower_bound[3]) / (upper_bound[3] - lower_bound[3]), 
                (s_T - lower_bound[4]) / (upper_bound[4] - lower_bound[4]), 
                (A - lower_bound[5]) / (upper_bound[5] - lower_bound[5]), 
                (b - lower_bound[6]) / (upper_bound[6] - lower_bound[6]), 
                (z_0 - lower_bound[7]) / (upper_bound[7] - lower_bound[7]), 
                (tau - lower_bound[8]) / (upper_bound[8] - lower_bound[8])]
    
    # Deviates the parameter by a small amount. The specific amount below was 
    # heuristical manner
    
    list_par[parameter] += random.gauss(0, 0.01)
    
    # Keeps the variations within the bounds
    
    if list_par[parameter] > 1:
        list_par[parameter] = 1
    if list_par[parameter] < 0:
        list_par[parameter] = 0
    
    # Rescales the deviated parameters to their normal values
            
    list_par = [(list_par[0] * (upper_bound[0] - lower_bound[0]) + lower_bound[0]), 
                (list_par[1] * (upper_bound[1] - lower_bound[1]) + lower_bound[1]), 
                (list_par[2] * (upper_bound[2] - lower_bound[2]) + lower_bound[2]), 
                (list_par[3] * (upper_bound[3] - lower_bound[3]) + lower_bound[3]), 
                (list_par[4] * (upper_bound[4] - lower_bound[4]) + lower_bound[4]), 
                (list_par[5] * (upper_bound[5] - lower_bound[5]) + lower_bound[5]), 
                (list_par[6] * (upper_bound[6] - lower_bound[6]) + lower_bound[6]), 
                (list_par[7] * (upper_bound[7] - lower_bound[7]) + lower_bound[7]), 
                (list_par[8] * (upper_bound[8] - lower_bound[8]) + lower_bound[8])]
    
# Checks if the deviated solution is within the preset boundaries and accepts 
# it if it is. We chose the d_L precent error to be less than 1 and the CMB
# percent error to be less than 10 due to the different scale of CMB fitnesses
# we have among our solutions (i.e. ~10-10000 for CMB to d_L) so that we have a 
# general method to get uncertanties.
    
    if (abs((GA.d_L_chi2_eval(list_par[5], list_par[6], list_par[7], list_par[8]) - d_L_fit) / d_L_fit * 100 < 1) 
    and (abs(GA.CMB_chi2_eval(list_par[0], list_par[1], list_par[2], list_par[3], 
                          list_par[4], list_par[5], list_par[6], list_par[7], 
                          list_par[8]) - CMB_fit) / CMB_fit * 100 < 10)):
        count += 1
        
        if list_par[parameter] < error_list[parameter][0]:
            error_list[parameter][0] = list_par[parameter]
        elif list_par[parameter] > error_list[parameter][1]:
            error_list[parameter][1] = list_par[parameter]
        
# Prints out the list containing the uncertanties for each parameter
            
print(error_list)

print()

print(count)

print()
    
print("Time elapsed: ", time.time() - start_time)
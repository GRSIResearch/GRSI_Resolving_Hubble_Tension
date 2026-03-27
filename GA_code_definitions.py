"""
Code that holds definitions for use in other codes, namely the plot code

Details: This just holds a bunch of definitions to be used in other codes
"""
import math
from math import pi, e, cos, sin, sqrt, exp, sinh, log, log10
import scipy.integrate as integrate
import numpy as np

# High-z Lmabda CDM constants
H_0_1 = 67.28     # (km/s)/Mpc
O_K_1 = 0.000001  # No curvature
O_M_1 = 0.31      # Matter density
O_R_1 = 0.0       # Negligible radiation density
O_L_1 = 0.69      # Dark energy density

# Low-z Lambda CDM constants
H_0_2 = 73.0     # (km/s)/Mpc
O_K_2 = 0.000001
O_M_2 = 0.31
O_R_2 = 0.0
O_L_2 = 0.69


# GR-SI Constants for Hubble tension definitions
h=.7306
Omega_M = 1.0
Omega_R = 0.0
Omega_L = 0.0

# Calculates the angular diameter distance at last scattering
 
def GRSI_d_a(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau): 
    speed_of_light = 299792458 # m/s
    h = .7306 # 0.71992 # Hubble parameter 
    H_0 = 100 * 3.2407792896664 * 10**(-20) * h # 1 / s
    proportion_matter = 1 # No dark matter
    proportion_radiation = 4.1192 * 10**(-5) / h**2 
    
    screened_proportion_curvature = 1 - proportion_radiation - depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau) * proportion_matter
    integrand, error = integrate.quad(lambda x: 1 / math.sqrt(screened_proportion_curvature * x**2 + 
    depletion_function(N, n_s, 1 / x - 1, proportion_baryon, sigma_temperature, A, b, z_0, tau) * x), 1 / (1 + z), 1)
    return (1 / (H_0 * (1 + z)) * integrand * speed_of_light)

# Defines the depletion function as seen in the hubble tension paper
 
def depletion_function(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    # Stops overflow error resulting in very large number in the denominator
    # of the middle term in depletion function
    
    try: 
        return 1 - 1 / (1 + math.exp((z - z_0) / tau)) + A * math.exp(-z / b)
    except OverflowError:
        return 1 + A * math.exp(-z / b)

# Calculates when the time of last scattering happens
 
def GRSI_t_l(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    h = .7306 # 0.71992 # Hubble parameter 
    H_0 = 100 * 3.2407792896664 * 10**(-20) * h # 1 / s
    proportion_matter = 1 # No dark matter
    proportion_radiation = 4.1192 * 10**(-5) / h**2 
    
    screened_proportion_curvature = 1 - proportion_radiation - depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau) * proportion_matter
    integrand, error = integrate.quad(lambda x: x / (math.sqrt(screened_proportion_curvature * x**2
    + depletion_function(N, n_s, 1 / x - 1, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    * x + proportion_radiation)), 0, 1 / (1 + z))
    return 1 / H_0 * integrand

# Calculates the term d_t, which is a value defined by having the arguments of 
# transfer functions being comoving wave numbers
 
def GRSI_d_t(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    speed_of_light = 299792458 # m/s
    h = .7306 # 0.71992 # Hubble parameter 
    H_0 = 100 * 3.2407792896664 * 10**(-20) * h # 1 / s
    proportion_radiation = 4.1192 * 10**(-5) / h**2 
    
    return (math.sqrt(proportion_radiation) / ((1 + z) * H_0 * 
    depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau)) * speed_of_light)

# Defines the term R for times of last scattering, the matter-radiation 
# equality, and the modern day
 
def GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    h = .7306 # 0.71992 # Hubble parameter 
    proportion_photon = 2.45 * 10**(-5) / h**2 
    return 3 * proportion_baryon / (4 * proportion_photon * (1 + z))
 
def GRSI_R_EQ(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    proportion_matter = 1 # No dark matter
    h = .7306
    proportion_radiation = 4.1192 * 10**(-5) / h**2 
    proportion_photon = 2.45 * 10**(-5) / h**2 
    
    return (3 * proportion_radiation * proportion_baryon / (4 * proportion_photon * proportion_matter
    * depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau)))
 
def GRSI_R_0(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    h = .7306 
    proportion_photon = 2.45 * 10**(-5) / h**2 
    return 3 / 4 * proportion_baryon / proportion_photon # * depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau)

# Calculates the acoustic horizon distance of the oscillations in the CMB
 
def GRSI_d_H(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    speed_of_light = 299792458 # m/s
    h = .7306 # 0.71992 # Hubble parameter 
    H_0 = 100 * 3.2407792896664 * 10**(-20) * h # 1 / s
    GRSI_RL = GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_EQ = GRSI_R_EQ(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    return (2 / (H_0 * math.sqrt(3 * GRSI_RL * depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau))
    * math.pow(1 + z, 1.5)) * math.log((math.sqrt(1 + GRSI_RL) + 
    math.sqrt(GRSI_RL + GRSI_EQ))/ (1 + math.sqrt(GRSI_EQ)), math.e) * speed_of_light)

# Calculates the effect of how the CMB photons are averaged as the event of 
# last scattering happens, more prominent for higher frequency waves
 
def GRSI_d_Landau_squared(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    speed_of_light = 299792458 # m/s
    T_0 = 2.72548 # K
    GRSI_tl = GRSI_t_l(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    return (3 * (sigma_temperature)**2 * GRSI_tl * GRSI_tl
    / (8 * (T_0 * (1 + z))**2 * (1 + GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)))
    * speed_of_light * speed_of_light)

# Fitting function used in X(T)

def g(T):
    return (84.2 * T**(-0.1166) / (1 + 5.085 * 10**(-3) * T**(0.53) + 4.22 * 10**4 * T**(0.8834) * math.exp(-39474 / T)))

# Calculates the fractional ionization of hydrogen in the early universe using
# Weinberg's approximation

def GRSI_X(T, N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    
    h = .7306
        
    integrand, error = integrate.quad(lambda temp: g(temp), T, 100000)
    return (1 / (1 + proportion_baryon * h * h / 
    math.sqrt(depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau) * h * h) * integrand))

# Calculates the effect of photons diffusing into nearby media and then reflecting
# out of the CMB, more present for higher frequency oscillations due to over
# and under dense regions being nearby
 
def GRSI_d_Silk_squared(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    
    GRSI_REQ = GRSI_R_EQ(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_R0 = GRSI_R_0(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_RL = GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    Y = .24 # Abundance of Helium to nucleons
    sigma_cross_section = 0.66525 * 10**(-24) / 100 / 100 # Thompson cross section, m^2
    G = 6.673 * 10**(-11) # SI, Newton's G from gravity
    speed_of_light = 299792458 # m/s
    m_n = 1.6726219 * 10**(-27) # kg, mass of a nucleon
    h = .7306 # 0.71992 # Hubble parameter 
    H_0 = 100 * 3.2407792896664 * 10**(-20) * h # 1 / s
    T_0 = 2.72548 # K
    
    integrand, error = integrate.quad(lambda R: (R**2 / (GRSI_X(T_0 * (1 + z) * GRSI_RL / R, N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    * (1 + R) * math.sqrt(GRSI_REQ + R)) * (16 / 15 + R**2 / (1 + R))), 0, GRSI_RL)
    return (GRSI_RL * GRSI_RL / (6 * (1 - Y) * (3 * H_0**2 * proportion_baryon
    / 8 / math.pi / G / m_n) * sigma_cross_section * math.sqrt(depletion_function(N, n_s, 0, proportion_baryon, sigma_temperature, A, b, z_0, tau)) 
    * H_0 * math.pow(GRSI_R0, 9 / 2)) * integrand * speed_of_light)

# Calculates the l distances for lambda CDM and GRSI
 
def GRSI_d_D(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    return (math.sqrt(GRSI_d_Landau_squared(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    + GRSI_d_Silk_squared(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)))

# Defines the l terms present in the power spectrum for lambda CDM and GRSI
 
def GRSI_l_R(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    k_R = 0.05 * (3.2407792896664 * 10**(-23)) # A scale length in meters^-1
    return (1 + z) * k_R * GRSI_d_a(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)

 
def GRSI_l_T(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    return (GRSI_d_a(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    / GRSI_d_t(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau))
 
def GRSI_l_D(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    return (GRSI_d_a(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    / GRSI_d_D(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau))

 
def GRSI_l_H(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    return (GRSI_d_a(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    / GRSI_d_H(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau))

# Defines the transfer functions used in the power spectrum

def transfer_T(k): 
    return (math.log(1 + (0.124 * k)**2, math.e) / (0.124 * k)**2 
    * (1 + (1.257 * k)**2 + (0.4452 * k)**4 + (0.2197 * k)**6)**(.5) 
    / (1 + (1.606 * k)**2 + (0.8568 * k)**4 + (0.3297 * k)**6)**(.5))

def transfer_S(k):
    return ((1 + (1.209 * k)**2 + (0.5116 * k)**4 + math.sqrt(5) * (0.1657 * k)**6)
    / (1 + (0.9459 * k)**2 + (0.4249 * k)**4 + (0.1657 * k)**6))**2

def transfer_delta(k):
    return (((0.1585 * k)**2 + (0.9702 * k)**4 + (.2460 * k)**6)
    / (1 + (1.180 * k)**2 + (1.540 * k)**4 + (0.9230 * k)**6 + (0.4197 * k)**8))**(0.25)

# Copmutes the power spectrum from GRSI for the given parameters
# Note that this is only to the first order, second order corrections are handled
# in the evaluator independently

def GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    
    l_values = [47.711224, 76.4716065, 105.917385, 135.605348, 165.405597, 195.26687, 225.164945, 255.086908, 285.025248, 314.975304, 344.934027, 374.899344, 404.869791, 434.844309, 464.822111, 494.802601, 524.78532, 554.769905, 584.756069, 614.743583, 644.732258, 674.721939, 704.712498, 734.703827, 764.695836, 794.688447, 824.681596, 854.675226, 884.669287, 914.663738, 944.658541, 974.653663, 1004.64908, 1034.64476, 1064.64068, 1094.63683, 1124.63318, 1154.62972, 1184.62643, 1214.62331, 1244.62034, 1274.61751, 1304.61481, 1334.61223, 1364.60976, 1394.6074, 1424.60514, 1454.60297, 1484.60089, 1514.5989, 1544.59698, 1574.59513, 1604.59335, 1634.59164, 1664.58999, 1694.58839, 1724.58686, 1754.58537, 1784.58394]
    
    # Reduces run time of code
    GRSI_LR = GRSI_l_R(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LD = GRSI_l_D(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LT = GRSI_l_T(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LH = GRSI_l_H(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_RL = GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    T_0 = 2.72548 # K
    tau_reionization = 0 # Assumes no blocking/dampening of outgoing light
    
    GRSI_list = [(4 * math.pi * T_0**2 * N * N * math.exp(-2 * tau_reionization)
                  / 25 * (10**6)**2) * integrate.quad(lambda B: math.pow(B * l_values[i]
                  / GRSI_LR, n_s - 1) * (3 * math.sqrt(B * B - 1) / 
                  (B**4 * (1 + GRSI_RL)**(1.5)) * (transfer_S(l_values[i] * B 
                  / GRSI_LT))**2 * math.exp(-2 * B**2 * l_values[i]**2 / (GRSI_LD)**2) 
                  * (math.sin(B * l_values[i] / GRSI_LH + transfer_delta(B * l_values[i]
                  / GRSI_LT)))**2 + 1 / (B**2 * math.sqrt(B**2 - 1)) * 
                  (3 * transfer_T(B * l_values[i] / GRSI_LT) * GRSI_RL - 
                  (1 + GRSI_RL)**(-.25)* transfer_S(B * l_values[i] / GRSI_LT)
                  * math.exp(-B**2 * l_values[i]**2 / (GRSI_LD)**2) * 
                  math.cos(B * l_values[i] / GRSI_LH + transfer_delta(B * l_values[i] / GRSI_LT)))**2)
                  , 1, math.inf, limit = 100)[0] for i in range(len(l_values))]
                                         
    return GRSI_list

def Individual_GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau, l):
        
    # Reduces run time of code
    GRSI_LR = GRSI_l_R(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LD = GRSI_l_D(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LT = GRSI_l_T(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_LH = GRSI_l_H(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    GRSI_RL = GRSI_R_L(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    T_0 = 2.72548 # K
    tau_reionization = 0 # Assumes no blocking/dampening of outgoing light
    
    GRSI_value = (4 * math.pi * T_0**2 * N * N * math.exp(-2 * tau_reionization)
                  / 25 * (10**6)**2) * integrate.quad(lambda B: math.pow(B * l
                  / GRSI_LR, n_s - 1) * (3 * math.sqrt(B * B - 1) / 
                  (B**4 * (1 + GRSI_RL)**(1.5)) * (transfer_S(l * B 
                  / GRSI_LT))**2 * math.exp(-2 * B**2 * l**2 / (GRSI_LD)**2) 
                  * (math.sin(B * l / GRSI_LH + transfer_delta(B * l
                  / GRSI_LT)))**2 + 1 / (B**2 * math.sqrt(B**2 - 1)) * 
                  (3 * transfer_T(B * l / GRSI_LT) * GRSI_RL - 
                  (1 + GRSI_RL)**(-.25)* transfer_S(B * l / GRSI_LT)
                  * math.exp(-B**2 * l**2 / (GRSI_LD)**2) * 
                  math.cos(B * l / GRSI_LH + transfer_delta(B * l / GRSI_LT)))**2)
                  , 1, math.inf, limit = 100)[0]
                                         
    return GRSI_value


# Calculates the expression inside of the integral for luminosity distance
# in Lambda CDM

def CDM_int(x,Omega_K,Omega_M,Omega_R,Omega_L):
    
    return 1.0/((x**2)*math.sqrt(Omega_K*(x**-2) + Omega_M*(x**-3) + Omega_R*(x**-4) + Omega_L))

# Calculates the expression inside of the integral for luminosity distance
# in GR-SI

def SI_int(x,Omega_M,Omega_R,Omega_L,A, b, z_0, tau):
    
    # Calculates the depletion function value
    
    D_Mv = depletion_function(0, 0, 1/x - 1, 0, 0, A, b, z_0, tau)
    
    # Gets the screened densities
    
    Omega_Mstar = D_Mv*Omega_M
    Omega_Rstar = D_Mv*Omega_R
    Omega_Lstar = D_Mv*Omega_L
    Omega_K = 1 - Omega_M * depletion_function(0, 0, 0, 0, 0, A, b, z_0, tau)
    
    # Returns the expression in the integral
    
    return 1.0/((x**2)*math.sqrt(Omega_K*(x**-2) + Omega_Mstar*(x**-3) + Omega_Rstar*(x**-4) + Omega_Lstar))

# Calculates the fitness of a solution to the power spectrum objective

def CMB_chi2_eval(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau):
    
    # Hard coded l_values, planck values, and planck error to reduce run time, 
    # found in Placnk 2018 data column 1,2, and 3 respectively
    
    l_values = [47.711224, 76.4716065, 105.917385, 135.605348, 165.405597, 195.26687, 225.164945, 255.086908, 285.025248, 314.975304, 344.934027, 374.899344, 404.869791, 434.844309, 464.822111, 494.802601, 524.78532, 554.769905, 584.756069, 614.743583, 644.732258, 674.721939, 704.712498, 734.703827, 764.695836, 794.688447, 824.681596, 854.675226, 884.669287, 914.663738, 944.658541, 974.653663, 1004.64908, 1034.64476, 1064.64068, 1094.63683, 1124.63318, 1154.62972, 1184.62643, 1214.62331, 1244.62034, 1274.61751, 1304.61481, 1334.61223, 1364.60976, 1394.6074, 1424.60514, 1454.60297, 1484.60089, 1514.5989, 1544.59698, 1574.59513, 1604.59335, 1634.59164, 1664.58999, 1694.58839, 1724.58686, 1754.58537, 1784.58394]
    planck_data = [1479.33552, 2034.96833, 2955.39416, 3869.51392, 4889.46506, 5464.10945, 5793.43954, 5372.88375, 4627.67753, 3604.23851, 2631.20029, 2033.05943, 1753.36253, 1787.57901, 2162.04649, 2422.0848, 2573.4805, 2546.29768, 2360.6453, 2095.43505, 1884.67698, 1813.16288, 1883.19393, 2097.13072, 2318.73584, 2464.58129, 2521.9126, 2394.12077, 2083.33867, 1740.72841, 1418.67342, 1172.95424, 1062.40462, 1047.86241, 1132.35191, 1211.01851, 1231.87224, 1205.7805, 1117.92662, 968.422243, 864.397967, 761.605936, 732.339633, 737.290337, 774.925851, 806.625346, 809.057064, 777.751089, 728.94358, 648.288645, 551.286111, 476.600995, 419.54127, 395.211496, 391.657847, 392.947519, 397.735803, 383.011743, 375.391143]
    planck_error = [50.7654876, 54.7101576, 64.976644, 76.9143744, 86.5856259, 90.5533459, 87.1348811, 76.9383697, 62.5199141, 47.1589439, 33.8762191, 24.8190562, 20.7109083, 20.6713695, 22.7523861, 24.9726104, 25.9184118, 25.0510538, 22.7209283, 19.8505488, 17.5025264, 16.4306044, 16.7990643, 18.1645476, 19.7363538, 20.706222, 20.519762, 19.0381031, 16.5790348, 13.7374917, 11.1420483, 9.28646475, 8.37987, 8.30074773, 8.68575434, 9.11274984, 9.26050628, 8.98528122, 8.33171031, 7.54508665, 6.64322644, 5.98585735, 5.68666086, 5.71800485, 5.94089388, 6.17243469, 6.25632882, 6.1114343, 5.74544667, 5.24258624, 4.72393254, 4.300796, 4.04442416, 3.96833591, 4.03423921, 4.17356282, 4.3166551, 4.41586492, 4.45733427]
    
    # The second order correction for hydrodynamic approximation
    
    C_L = [-45.576530077465804, -262.5124188797224, -149.13532372990767, -52.79646198254295, 229.79871927776458, 298.1731059771646, 490.68734992510326, 358.6297396715818, 266.6266413942076, 97.06811789214817, -36.53649472549614, -5.2859900716150605, 20.66442460595499, 36.39592063008604, 171.10420872710483, 129.99960298996984, 72.24008988910555, 18.502985186470596, -11.406947124180988, -20.869022692546423, 0.16833557016866507, 25.107956663137657, 3.7347937374529465, -34.39932347332842, -129.96863678708905, -239.9188754055549, -267.4932410020933, -256.4057932402434, -225.27980169077318, -106.8758056226261, 36.39890285604383, 155.41851367017193, 247.79863814353416, 272.1670018002193, 281.45424776156847, 248.46585063100906, 195.43269204338867, 178.00386723251336, 184.7247895572698, 182.39971330896822, 225.92515714453486, 222.5401806704058, 218.34121113822664, 178.26479681086505, 131.32232694015636, 82.25309605471682, 47.100763652593514, 43.17875819892106, 85.05656395855578, 136.33668252337498, 179.86586987002164, 223.21885562586814, 242.11051511447616, 248.08145619240517, 239.98346801141784, 220.74515794298057, 207.9369157464186, 190.93433984754367, 199.15344952040974]
    
    # Calculates the fitness of a genetic algorithm solution with respect
    # to the power spectrum objective using the fitness function
    
    chi_squared_CMB = 0 # Fitness parameter
    
    # Calculates the first order GR-SI power spectrum for a given solution
    
    GRSI_list = GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    # Calculates the fitness objective for each parameter. Note that here
    # we include the second order corrections in the model data point 
    # implicitly
    
    for i in range(len(l_values)):
        chi_squared_CMB += ((GRSI_list[i] + C_L[i]) - planck_data[i])**2 / (planck_error[i])**2 / (len(l_values) - 9) * max(planck_data) # norm_factor
    return chi_squared_CMB

# Calculates the fitness of a solution to the Hubble tension objective

def d_L_chi2_eval(A, b, z_0, tau):
    
    h = .7306
    
    # Gets the luminosity data at low-z and high-z with errors
    # From:
    # Conley et al. (2011) arXiv:1104.1443
    # Kowalski et al. (2008) arXiv:0804.4142
    # Riess et al. (2007)+Essence
    # Schaefer astro-ph/0201196
    # Suzuki et al. (2012), arXiv:1105.3470
    # Aghanim, N.; et al. for CMB data point
    
    z_L = [0.01376, 0.0197, 0.02603, 0.03432, 0.05863, 0.11585, 0.17641, 0.23664, 0.28563, 0.3574, 0.44228, 0.52328, 0.58521, 0.64596, 0.71506, 0.76831, 0.83455, 0.91428, 1.02478, 1.32375, 0.01165, 0.03231, 0.10263, 0.27094, 0.36235, 0.44353, 0.51734, 0.60119, 0.69209, 0.80419, 0.90584, 0.99577, 1.1475, 1.275, 1.36667, 1.551, 0.01536, 0.02986, 0.06467, 0.216, 0.33167, 0.43777, 0.50223, 0.59268, 0.69855, 0.81217, 0.91862, 0.99863, 1.14975, 1.26625, 1.36667, 1.755, 0.17, 0.25, 0.44, 0.68, 0.82857, 1.02, 1.37625, 1.602, 2.17375, 2.65857, 3.3, 4.10429, 4.9, 6.445, 0.0068, 0.0117, 0.0154, 0.0192, 0.0237, 0.0284, 0.0331, 0.0436, 0.0668, 0.1101, 0.1581, 0.1991, 0.2433, 0.279, 0.3172, 0.3564, 0.3998, 0.4383, 0.4876, 0.5435, 0.5989, 0.6671, 0.7833, 0.8912, 1.0386, 1.2949, 1089.0]
    d_L = [0.00018979872391463535, 0.00027492879960197705, 0.00036877985066608446, 0.000503446924548382, 0.0008493864641367975, 0.0017148575827477647, 0.0027276651619832586, 0.0037779080928882097, 0.004624203007524841, 0.006158783982101135, 0.007816636173851541, 0.009633558206285416, 0.01122676701744518, 0.012349689671350118, 0.013936350497916998, 0.015302357769693482, 0.017097933665127664, 0.01888850735888032, 0.0228523874122303, 0.03149934097769502, 0.00016096278999675948, 0.0004512882814031318, 0.0015086355233416126, 0.004480494315462888, 0.006099576070050071, 0.007972744012482115, 0.009521932087655423, 0.011514125744212783, 0.013242783931920438, 0.01608841457467306, 0.01870209833719458, 0.02139019036015702, 0.022027518629040885, 0.02792727961510812, 0.030500808708495173, 0.037022830125279224, 0.00021385085676112116, 0.00042366543487538916, 0.0009309806460352573, 0.003489720554214434, 0.0054754575958619696, 0.007940463775154886, 0.009489799177452303, 0.011067673375975126, 0.013476437778888791, 0.01595054984447595, 0.018482494896027626, 0.021616960771870503, 0.02426982512562214, 0.029076091395826124, 0.028943520486553093, 0.034251074968642124, 0.0031131195547689093, 0.004733579451512653, 0.008088530864719844, 0.015368477688192573, 0.018954104900984075, 0.020106397254816107, 0.031088450268530844, 0.03449636738756835, 0.06040364844833285, 0.06513101849710567, 0.09694757575382258, 0.11457776366125219, 0.1816319790369804, 0.22458964041888582, 9.371306651820153e-05, 0.00016289067459059934, 0.0002111295972657784, 0.00026392936354795724, 0.000326889990044141, 0.0004026234687575545, 0.00046297109617206186, 0.0005967460149637066, 0.0009190561796727708, 0.0016058223592345447, 0.0023279335473665595, 0.00308037352997769, 0.003831039320778245, 0.004420136413408467, 0.005130549034664711, 0.005946396355843476, 0.0070091440083356405, 0.0076645339723658305, 0.008498286912354729, 0.009640031091330091, 0.011035474877500593, 0.012247780211177372, 0.015525131605857835, 0.017256807607081752, 0.021674495422115463, 0.02955871930609908, 51.074151164]
    e_L = [2.534760731659765e-06, 3.2412004126630898e-06, 4.160820235027452e-06, 5.332455164462493e-06, 8.761915055095156e-06, 1.4451896154312676e-05, 2.6504474413592973e-05, 3.566571491528242e-05, 4.4081150576114417e-05, 4.708133230467986e-05, 6.119479776631316e-05, 8.163008206728239e-05, 0.00010598740414523464, 0.00013080657216562528, 0.00014953768934775625, 0.00019379239488486805, 0.0002275554046907128, 0.00024181771883247186, 0.0003978039234561306, 0.0012257555292669723, 5.025749861685534e-06, 7.107646938707993e-06, 1.9522540567459095e-05, 9.099346528465938e-05, 9.157204991977286e-05, 0.00013658293605968747, 0.0001561064192365438, 0.00021633999506746396, 0.00030431651597638315, 0.0003963808961941886, 0.0006924558192500423, 0.0008323711954672822, 0.0014668292260129974, 0.0017143696398717878, 0.0026252238447218752, 0.006819857339753507, 1.5747265240971158e-05, 1.740337875133465e-05, 2.7224509399319725e-05, 0.00014656530432972714, 0.00013893693155383173, 0.00016308965419956732, 0.0002019038879227416, 0.00021253872614525942, 0.0003767120270990598, 0.00044587182932495867, 0.0006528323142918131, 0.0009078940233156247, 0.0014227897738659087, 0.0015237859741363835, 0.0018127417916620537, 0.005522198345220017, 0.0005466486615401306, 0.0024855150205740227, 0.0014799052009456962, 0.0016752313558547863, 0.0018469903552408916, 0.001983350224993666, 0.0026715074963352507, 0.003652229163790618, 0.0056190154353752, 0.007834417769661569, 0.009906949296475577, 0.012141228828381631, 0.07946238659489602, 0.04786617832690548, 2.8785360151808266e-06, 3.16558775400449e-06, 3.28633251627316e-06, 3.536929341393987e-06, 3.838729292435516e-06, 4.542666506528425e-06, 4.882418977960693e-06, 5.743564439540757e-06, 7.99925512278783e-06, 1.3163251749877171e-05, 2.0583417921204434e-05, 2.75201500234027e-05, 3.934297137691318e-05, 4.559627616102498e-05, 5.765000554333851e-05, 6.626968465303664e-05, 0.00010555004432390386, 0.00010024201268285087, 0.00011114640335892659, 0.00012208345537792687, 0.00014382127890026427, 0.00018218205264194823, 0.0002116277846863203, 0.00027576275956420486, 0.0003713108332250529, 0.0006084695099934383, 0.068486]

    Omega_M = 1.0 
    Omega_R = 0.0 # Radiation is negligible
    Omega_L = 0.0 # No dark matter

    chi_SI = 0.0 # Fitness parameter
    
    # Gets the fitness value for each data points and returns it

    for i in range(len(d_L)):
        
        z = z_L[i]
        
        arg_SI = integrate.quad(SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A, b, z_0, tau))[0]
        D_L_SI = (((1+z)/(100 * h))*arg_SI)  # 100 h for H_0 in km / Mpc / s
        chi = ((D_L_SI - d_L[i]) / (e_L[i]))**2 * max(d_L) # norm_factor
        chi_SI = chi_SI + chi

    chi_SI = chi_SI/(len(d_L)-4)  # SI: ndf = N - 4 parameters of D_M
    return chi_SI

# Calculates the r resdiues for the luminositiy distance data for each model
# and their true reduced chi values. Only used for the plotting, not for the GA

def chi2_3(A, b, z_0, tau):
    
    chi_LCDM_SN = 0.0
    chi_LCDM_CMB = 0.0
    chi_SI = 0.0
    r_CMB = []
    r_SN = []
    r_SI = []
    
    z_L,d_L,e_L,z_L_C,d_L_C,e_L_C,z_L_K,d_L_K,e_L_K,z_L_R,d_L_R,e_L_R,z_L_Sc,d_L_Sc,e_L_Sc,z_L_Su,d_L_Su,e_L_Su,z_L_CMB,d_L_CMB,e_L_CM = read_data()
    
    norm_factor = max(d_L)

    for i in range(len(d_L)):
        
        # High-z Lambda CDM
        z = z_L[i]
        arg_CDM_1 = integrate.quad(CDM_int, 1/(1+z), 1, args=(O_K_1,O_M_1,O_R_1,O_L_1))[0]
        D_L_CDM_1 = (((1+z)/(H_0_1*sqrt(O_K_1)))*sinh(sqrt(O_K_1)*arg_CDM_1))
        chi_CMB = ((D_L_CDM_1 - d_L[i])/e_L[i])**2
        chi_LCDM_CMB = chi_LCDM_CMB + chi_CMB
        r_CMB.append(chi_CMB)
        
        # Low-z Lambda CDM
        arg_CDM_2 = integrate.quad(CDM_int, 1/(1+z), 1, args=(O_K_2,O_M_2,O_R_2,O_L_2))[0]
        D_L_CDM_2 = (((1+z)/(H_0_2*sqrt(O_K_2)))*sinh(sqrt(O_K_2)*arg_CDM_2))
        chi_SN = ((D_L_CDM_2 - d_L[i])/e_L[i])**2
        chi_LCDM_SN = chi_LCDM_SN + chi_SN
        r_SN.append(chi_SN)
        
        # 100 h for H_0 in km / Mpc / s
        
        # GR-SI
        arg_SI = integrate.quad(SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A, b, z_0, tau))[0]
        D_L_SI = (((1+z)/(100 * h))*arg_SI)
        chi = ((D_L_SI - d_L[i]) / (e_L[i]))**2  / norm_factor 
        chi_SI = chi_SI + chi
        r_SI.append(chi * norm_factor) # Counter the norm factor

    chi_LCDM_CMB = chi_LCDM_CMB/len(d_L)
    chi_LCDM_SN = chi_LCDM_SN/len(d_L)
    chi_SI = chi_SI/(len(d_L)-4)         # SI: ndf = N - 4 parameters of D_M
    return chi_LCDM_CMB, chi_LCDM_SN, chi_SI, r_CMB, r_SN, r_SI

# Gets the luminosity distance data for the Hubble tension objective
# (where the hard coded z_L, d_L, and e_L come from)

def read_data():
    z_L = []
    d_L = []
    e_L = []

    d_C  = np.loadtxt('data_Conley.txt')    # Conley et al. (2011) arXiv:1104.1443 
    z_L_C = []
    d_L_C = []
    e_L_C = []
    for i in range(len(d_C[:,0])):
        z_L_C.append(d_C[i,0])
        d_L_C.append(dL(d_C[i,0],d_C[i,1],H_0_2))
        e_L_C.append(error(d_C[i,0],d_C[i,1],d_C[i,2],H_0_2))
        z_L.append(d_C[i,0])
        d_L.append(dL(d_C[i,0],d_C[i,1],H_0_2))
        e_L.append(error(d_C[i,0],d_C[i,1],d_C[i,2],H_0_2))
    
    d_K  = np.loadtxt('data_Kowalski.txt')  # Kowalski et al. (2008) arXiv:0804.4142
    z_L_K = []
    d_L_K = []
    e_L_K = []
    for i in range(len(d_K[:,0])):
        z_L_K.append(d_K[i,0])
        d_L_K.append(dL(d_K[i,0],d_K[i,1],H_0_2))
        e_L_K.append(error(d_K[i,0],d_K[i,1],d_K[i,2],H_0_2))
        z_L.append(d_K[i,0])
        d_L.append(dL(d_K[i,0],d_K[i,1],H_0_2))
        e_L.append(error(d_K[i,0],d_K[i,1],d_K[i,2],H_0_2))
    
    d_R  = np.loadtxt('data_Riess.txt')     # Riess et al. (2007)+Essence
    z_L_R = []
    d_L_R = []
    e_L_R = []
    for i in range(len(d_R[:,0])):
        z_L_R.append(d_R[i,0])
        d_L_R.append(dL(d_R[i,0],d_R[i,1],H_0_2))
        e_L_R.append(error(d_R[i,0],d_R[i,1],d_R[i,2],H_0_2))
        z_L.append(d_R[i,0])
        d_L.append(dL(d_R[i,0],d_R[i,1],H_0_2))
        e_L.append(error(d_R[i,0],d_R[i,1],d_R[i,2],H_0_2))
    
    d_Sc = np.loadtxt('data_Schaefer.txt')  # Schaefer astro-ph/0201196
    z_L_Sc = []
    d_L_Sc = []
    e_L_Sc = []
    for i in range(len(d_Sc[:,0])):
        z_L_Sc.append(d_Sc[i,0])
        d_L_Sc.append(dL(d_Sc[i,0],d_Sc[i,1],H_0_2))
        e_L_Sc.append(error(d_Sc[i,0],d_Sc[i,1],d_Sc[i,2],H_0_2))
        z_L.append(d_Sc[i,0])
        d_L.append(dL(d_Sc[i,0],d_Sc[i,1],H_0_2))
        e_L.append(error(d_Sc[i,0],d_Sc[i,1],d_Sc[i,2],H_0_2))
    
    d_Su = np.loadtxt('data_Suzuki.txt')    # Suzuki et al. (2012), arXiv:1105.3470
    z_L_Su = []
    d_L_Su = []
    e_L_Su = []
    for i in range(len(d_Su[:,0])):
        z_L_Su.append(d_Su[i,0])
        d_L_Su.append(dL(d_Su[i,0],d_Su[i,1],H_0_2))
        e_L_Su.append(error(d_Su[i,0],d_Su[i,1],d_Su[i,2],H_0_2))
        z_L.append(d_Su[i,0])
        d_L.append(dL(d_Su[i,0],d_Su[i,1],H_0_2))
        e_L.append(error(d_Su[i,0],d_Su[i,1],d_Su[i,2],H_0_2))

# CMB data point
    z_L_CMB = [1089.0]
    d_L_CMB = [51.074151164]
#   e_L_CMB = [0.06742]          # inverse weight for the CMB data point
#   e_L_CMB = [0.0672]           # inverse weight for the CMB data point
    e_L_CMB = [0.068486]           # inverse weight for the CMB data point
    z_L.append(z_L_CMB[0])
    d_L.append(d_L_CMB[0])
    e_L.append(e_L_CMB[0])    
 
    return z_L,d_L,e_L,z_L_C,d_L_C,e_L_C,z_L_K,d_L_K,e_L_K,z_L_R,d_L_R,e_L_R,z_L_Sc,d_L_Sc,e_L_Sc,z_L_Su,d_L_Su,e_L_Su,z_L_CMB,d_L_CMB,e_L_CMB

# Converts SN and gamma-ray data into D_L

def dL(z,ddm,h0):
    return exp(ddm/5.0*log(10.0))*(z+z**2.0/2.0)/h0

# Compute the error for the data 

def error(z,ddm,sig,h0):
    dm = exp(ddm/5.0*log(10.0))*(z+z**2.0/2.0)
    return (dm*sig*log(10.0)/5.0)/h0

if __name__ == "__main__":
    print(depletion_function(1.17907*10**(-5), 0.96635, 0, 0.041476, 430.83744, 0.26447, 0.21895, 12.06115, 0.13552))

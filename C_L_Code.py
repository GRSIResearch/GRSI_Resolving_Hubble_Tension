"""
Code that gets the second order corrections for hydrodynamic approximation

Details: To get the second order corrections, we use modern parameters with the
hydrodynamic approximation from S. Navas et al. “Review of particle physics”
and then find the difference between these values with the observed Planck
values
"""

# Imports
import math
import matplotlib.pyplot as py
import scipy.integrate as integrate



# S. Navas et al. “Review of particle physics”

# Constants across equations: 
# To get observational data, go to planck legacy archive
h = 0.6745 # Hubble parameter 
H_0 = 100 * 3.24078 * 10**(-20) * h # 1 / s
z_L = 1089 # Unitless (Also 1 + z_l = 1090)
proportion_matter = 0.1428 / h**2  # Unitless 0.13299 / h**2
proportion_radiation = 4.15 * 10**(-5) / h**2 # Unitless
proportion_dark_energy = 0.6857 # Unitless
proportion_baryon = 0.0223715 / h**2 # Unitless 0.02238 / h**2
proportion_photon = 2.473 * 10**(-5) / h**2 # Unitless
proportion_curvature = 0.0007 # Unitless
sigma_temperature = 262 # K (Temperature deviation at T_L)
T_L = 2.725 * (z_L + 1) # K
Y = .244833 # Abundance of Helium to nucleons, unitless
sigma_cross_section =  0.66525 * 10**(-28) # Thompson cross section, m^2
R_0 = 3 * proportion_baryon / 4 / proportion_photon # Unitless
rho_B_0 = proportion_baryon * 3 * H_0**2 / (8 * math.pi * 6.67 * 10**(-11))  # kg / m^3, also don't need I think
n_B_0 = 2.51517 * 10**(-7)*100**3 # 1 / m^3
rho_photon_0 = proportion_photon * H_0**2 / (8 * math.pi * 6.67 * 10**(-11)) # kg / m^3 (Might want to change this), also don't need I think
n_s = 0.9654 # Scalar spectral index, unitless
k_R = 0.05 * (3.24078 * 10**(-23)) # A scale length in meters
T_0 = 2.72556 # K, temperature of CMB today
N_squared = (1.2455 * 10**(-5))**(2) # Normalization value for primordial perturbations, calculated using N^2e^(-2*tau_reion) as constant between Wienberg and modern values
tau_reionization = 0.0547 # Optical depth at reionization
speed_of_light = 299792458 # m/s
R_T_conversion = 1852.82674 # Value by fitting R_x to temperature values for 0, L, EQ

# Here are each of the functions we need to compute the power spectrum values

# d_a is the angular diameter distance of an object in the sky, this means it
# is is the distance that is needed for an object of a specific size to span 
# a specific angle in the sky, which is made complicated by the fact that the 
# scale of the universe is changed. The path of the photons emitted by an object
# far in the past is curved by this expansion such that the angle between them
# now is greater than it was in the past, which increases the "size" of the 
# object from our perspective.

# lambda-CDM d_a

def CDM_d_a(): 
    integrand, error = integrate.quad(lambda x: 1 / math.sqrt(proportion_dark_energy * x**4 + proportion_curvature * x**2 + proportion_matter * x), 1 / (1 + z_L), 1)
    return 1 / (math.sqrt(proportion_curvature) * H_0 * (1 + z_L)) * math.sinh(math.sqrt(proportion_curvature) * integrand) * speed_of_light
# t(x) is the age of the universe with t_l being when the era of last scattering 
# happened

# lambda_CDM t_l

def CDM_t_l(): 
    integrand, error = integrate.quad(lambda x: x / (math.sqrt(proportion_dark_energy * x**4 + proportion_matter * x + proportion_radiation)), 0, 1 / (1 + z_L))
    return 1 / H_0 * integrand

# d_t is the distance associated with the scale of the adjusted wave numbers
# in the transfer functions that connect the analytic solutions of the 
# fluctuations across the matter-radiation equality using the scale of the 
# universe at the time of last scattering 

# lambda CDM d_t

def CDM_d_t():
    return math.sqrt(proportion_radiation) / ((1 + z_L) * H_0 * proportion_matter) * speed_of_light 

# All of the R's are computationally 3/4's of the ratio between the energy density 
# of baryons versus photons. It is a useful parameter to have as it helps 
# quantify interactions between both species.

# R_L (It is the same for both lambda CDM and GRSI)

def R_L():
    return 3 * proportion_baryon / (4 * proportion_photon * (1 + z_L))

# lambda CDM R_EQ

def CDM_R_EQ():
    return 3 * proportion_radiation * proportion_baryon / (4 * proportion_matter * proportion_photon)

# d_H is the acoustic horizon distance at last scattering, which means the 
# distance that a pressure wave extends throughout the plasma medium at last
# scattering

# lambda CDM d_H

def CDM_d_H():
    return (2 / (H_0 * math.sqrt(3 * R_L() * proportion_matter) * math.pow(1 + z_L, 1.5)) 
    * math.log((math.sqrt(1 + R_L()) + math.sqrt(R_L() + CDM_R_EQ())) / (1 + math.sqrt(CDM_R_EQ())), math.e)
    * speed_of_light)

# This is the distance associated in part with faster oscillatory nodes being
# reduced by the fact that the last scattering happens over a period of time.
# This happens because the fast nodes will more quickly put their energy in the
# dispersing plasma around last scattering than the slower nodes, so their effect
# on the observed power spectrum is lower.

# lambda CDM d_Landau^2

def CDM_d_Landau_squared():
    return (3 * sigma_temperature**2 * CDM_t_l() * CDM_t_l() / (8 * T_L**2 * (1 + R_L()))
    * speed_of_light * speed_of_light)

# d_Silk is another distance associated with the dampening caused by the photons
# being released and interacting with other matter throughout the last scattering
# The photons that come from over dense and under dense regions will mix through
# scattering, which happens more readily with higher oscillation numbers

# lambda CDM d_Silk^2

def CDM_d_Silk_squared():
    # I have divided by the conversion term here, however it might be better to
    # move it to the return line with the rest of the constants outfront.
    # I have also divided by the R_T_coversion term to get it into the right units
    # however this as far as I can tell does very little
    integrand, error = integrate.quad(lambda R: (R**2 / (CDM_X(R_T_conversion / R) * (1 + R) 
    * math.sqrt(CDM_R_EQ() + R)) * (16 / 15 + R**2 / (1 + R))), 0, R_L())
    return (R_L() * R_L() / (6 * (1 - Y) * n_B_0 * sigma_cross_section * math.sqrt(proportion_matter) 
    * H_0 * math.pow(R_0, 9 / 2)) * integrand * speed_of_light)

# This is in essense the combination of both Landau and Silk dampening that
# reduces the prominence of higher order oscillations in the CMB

# lambda CDM d_D

def CDM_d_D():
    return math.sqrt(CDM_d_Landau_squared() + CDM_d_Silk_squared())

# X is the fractional ionization at a specific temperature

# lambda CDM X

def CDM_X(T):
    integrand, error = integrate.quad(lambda temp: g(temp), T, 3400)
    return (1 / (1 / .437 + proportion_baryon * h * h / 
    math.sqrt(proportion_matter * h * h) * integrand))

# A fitting function that is integral to the calculation of X(T)

# g function from Wienberg

def g(T):
    return (84.2 * T**(-0.1166) / (1 + 5.085 * 10**(-3) * T**(0.53) + 4.22 * 10**4 * T**(0.8834) * math.exp(-39474 / T)))

# For convienience in the final computation, we introduce a few more terms/functions

# lambda CDM l_T

def CDM_l_T():
    return CDM_d_a() / CDM_d_t()

# lambda CDM l_D

def CDM_l_D():
    return CDM_d_a() / CDM_d_D()

# lambda CDM l_H

def CDM_l_H():
    return CDM_d_a() / CDM_d_H()

# lambda CDM l_R

def CDM_l_R():
    return (1 + z_L) * k_R * CDM_d_a()

# Our Transfer Functions

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

# The lambda CDM Power Spectrum

def CDM_power_spectrum(l):

    
    CDM_LR = CDM_l_R()
    CDM_LD = CDM_l_D()
    CDM_LT = CDM_l_T()
    CDM_LH = CDM_l_H()
    CDM_RL = R_L()
    
    integrand, error = integrate.quad(lambda B: math.pow(B * l / CDM_LR, n_s - 1) * (3 * 
    math.sqrt(B * B - 1) / (B**4 * (1 + CDM_RL)**(1.5)) * (transfer_S(l * B / CDM_LT))**2 
    * math.exp(-2 * B**2 * l**2 / (CDM_LD)**2) * (math.sin(B * l / CDM_LH + transfer_delta(B * l / CDM_LT)))**2
    + 1 / (B**2 * math.sqrt(B**2 - 1)) * (3 * transfer_T(B * l / CDM_LT) * CDM_RL
    - (1 + CDM_RL)**(-.25) * transfer_S(B * l / CDM_LT) * math.exp(-B**2 * l**2 / (CDM_LD)**2)
    * math.cos(B * l / CDM_LH + transfer_delta(B * l / CDM_LT)))**2), 1, math.inf)
    return ((4 * math.pi * T_0**2 * N_squared * math.exp(-2 * tau_reionization) 
    / 25 * integrand) * (10**6)**2) # To turn K into uK
    

if __name__ == "__main__":
    
    # Gets the CMB data from WMAP
    file = open('Planck_Data.txt')
    lines = file.readlines()
    file.close()
    
    # Gets the relevant data for plotting, which is the mean l number, the mean
    # TT value in the binned l number, and the associated error with a specific
    # l value due to cosmic variance or measurement errors
    
    l_values = []
    power_spectrum_values = []
    error_values = []

    for i in range(len(lines)):
        data = lines[i].split()
        if float(data[0]) < 1800:
            l_values.append(float(data[0]))
            power_spectrum_values.append(float(data[1]))
            error_values.append(float(data[2]))
        else:
            break

    
    # Calculates the CMB power spectrum using the data
    
    CDM_power_spectrum_values = []
    
    for i in range(len(l_values)):
        CDM_power_spectrum_values.append(CDM_power_spectrum(l_values[i]))
        
    # Plots just the first order part of power spectrum against observed data
        
    py.errorbar(l_values, power_spectrum_values, error_values, fmt = "o", color = "Black", capsize = 5, markersize = 5, label = "Planck 2018 Data")
    py.scatter(l_values, CDM_power_spectrum_values, color = "Blue", label = "λ-CDM Model")
    py.title("Power Spectrum")
    py.ylabel("l(l + 1)C / 2π")
    py.xlabel("l")
    py.legend(loc = "upper right")
    py.show()
    
    # Gets the second order corrections and outputs them
    
    correction_values = []
    for i in range(len(CDM_power_spectrum_values)):
        correction_values.append(power_spectrum_values[i] - CDM_power_spectrum_values[i])
    
    print(correction_values)
    
    # Gets the corrected power spectrum values
    
    CDM_power_spectrum_values = []
    
    for i in range(len(l_values)):
        CDM_power_spectrum_values.append(CDM_power_spectrum(l_values[i]) + correction_values[i])
        
    # Plots the corrected power spectrum value
    
    py.errorbar(l_values, power_spectrum_values, error_values, fmt = "o", color = "Black", capsize = 5, markersize = 5, label = "Planck 2018 Data")
    py.scatter(l_values, CDM_power_spectrum_values, color = "Blue", label = "λ-CDM Model")
    py.title("Corrected Power Spectrum")
    py.ylabel("l(l + 1)C / 2π")
    py.xlabel("l")
    py.legend(loc = "upper right")
    py.show()

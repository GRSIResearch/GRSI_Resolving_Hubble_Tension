'''
Code that produces the solution plots in the paper

Details: Gets the Pareto front information from the 6 runs of the genetic 
algorithm and with it, gets the meta front and the solutions used on the meta
front. 
'''

import math
import matplotlib.pyplot as py
import scipy.integrate as integrate
import numpy as np
import GA_code_definitions as GA
from scipy.interpolate import make_interp_spline

# Constants and data used for the plots

# Planck 2018 power spectrum data

l_values = [47.711224, 76.4716065, 105.917385, 135.605348, 165.405597, 195.26687, 225.164945, 255.086908, 285.025248, 314.975304, 344.934027, 374.899344, 404.869791, 434.844309, 464.822111, 494.802601, 524.78532, 554.769905, 584.756069, 614.743583, 644.732258, 674.721939, 704.712498, 734.703827, 764.695836, 794.688447, 824.681596, 854.675226, 884.669287, 914.663738, 944.658541, 974.653663, 1004.64908, 1034.64476, 1064.64068, 1094.63683, 1124.63318, 1154.62972, 1184.62643, 1214.62331, 1244.62034, 1274.61751, 1304.61481, 1334.61223, 1364.60976, 1394.6074, 1424.60514, 1454.60297, 1484.60089, 1514.5989, 1544.59698, 1574.59513, 1604.59335, 1634.59164, 1664.58999, 1694.58839, 1724.58686, 1754.58537, 1784.58394]
planck_data = [1479.33552, 2034.96833, 2955.39416, 3869.51392, 4889.46506, 5464.10945, 5793.43954, 5372.88375, 4627.67753, 3604.23851, 2631.20029, 2033.05943, 1753.36253, 1787.57901, 2162.04649, 2422.0848, 2573.4805, 2546.29768, 2360.6453, 2095.43505, 1884.67698, 1813.16288, 1883.19393, 2097.13072, 2318.73584, 2464.58129, 2521.9126, 2394.12077, 2083.33867, 1740.72841, 1418.67342, 1172.95424, 1062.40462, 1047.86241, 1132.35191, 1211.01851, 1231.87224, 1205.7805, 1117.92662, 968.422243, 864.397967, 761.605936, 732.339633, 737.290337, 774.925851, 806.625346, 809.057064, 777.751089, 728.94358, 648.288645, 551.286111, 476.600995, 419.54127, 395.211496, 391.657847, 392.947519, 397.735803, 383.011743, 375.391143]
planck_error = [50.7654876, 54.7101576, 64.976644, 76.9143744, 86.5856259, 90.5533459, 87.1348811, 76.9383697, 62.5199141, 47.1589439, 33.8762191, 24.8190562, 20.7109083, 20.6713695, 22.7523861, 24.9726104, 25.9184118, 25.0510538, 22.7209283, 19.8505488, 17.5025264, 16.4306044, 16.7990643, 18.1645476, 19.7363538, 20.706222, 20.519762, 19.0381031, 16.5790348, 13.7374917, 11.1420483, 9.28646475, 8.37987, 8.30074773, 8.68575434, 9.11274984, 9.26050628, 8.98528122, 8.33171031, 7.54508665, 6.64322644, 5.98585735, 5.68666086, 5.71800485, 5.94089388, 6.17243469, 6.25632882, 6.1114343, 5.74544667, 5.24258624, 4.72393254, 4.300796, 4.04442416, 3.96833591, 4.03423921, 4.17356282, 4.3166551, 4.41586492, 4.45733427]

# Second order corrections to the power spectrum

C_L = [-45.576530077465804, -262.5124188797224, -149.13532372990767, -52.79646198254295, 229.79871927776458, 298.1731059771646, 490.68734992510326, 358.6297396715818, 266.6266413942076, 97.06811789214817, -36.53649472549614, -5.2859900716150605, 20.66442460595499, 36.39592063008604, 171.10420872710483, 129.99960298996984, 72.24008988910555, 18.502985186470596, -11.406947124180988, -20.869022692546423, 0.16833557016866507, 25.107956663137657, 3.7347937374529465, -34.39932347332842, -129.96863678708905, -239.9188754055549, -267.4932410020933, -256.4057932402434, -225.27980169077318, -106.8758056226261, 36.39890285604383, 155.41851367017193, 247.79863814353416, 272.1670018002193, 281.45424776156847, 248.46585063100906, 195.43269204338867, 178.00386723251336, 184.7247895572698, 182.39971330896822, 225.92515714453486, 222.5401806704058, 218.34121113822664, 178.26479681086505, 131.32232694015636, 82.25309605471682, 47.100763652593514, 43.17875819892106, 85.05656395855578, 136.33668252337498, 179.86586987002164, 223.21885562586814, 242.11051511447616, 248.08145619240517, 239.98346801141784, 220.74515794298057, 207.9369157464186, 190.93433984754367, 199.15344952040974]

# Constants for high-z Hubble tension Lambda CDM data

H_0_1 = 67.28     # (km/s)/Mpc
O_K_1 = 0.000001
O_M_1 = 0.31
O_R_1 = 0.0
O_L_1 = 0.69

# Constants for low-z Hubble tension Lambda CDM data

H_0_2 = 73.06     # (km/s)/Mpc
O_K_2 = 0.000001
O_M_2 = 0.31
O_R_2 = 0.0
O_L_2 = 0.69

# Constants for GR-SI Hubble tension data

H_0_GRSI = 73.06 # (km/s)/Mpc 
Omega_M = 1.0
Omega_R = 0.0
Omega_L = 0.0

# Puts everything in Latex Font

py.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.formatter.use_mathtext": True # Fixes minus sign glyph issues
})

file = open("pymoo_Results", "r") # Put file from pymoo_GA.py here
lines = file.readlines()
file.close()

# Gets each individual element from the solution file i.e. the fitness and
# parameter values

chi_d_L = []
chi_CMB = []
N = []
n_s = []
z = []
prop_baryon = []
sigma_temp = []
A = []
b = []
z_0 = []
tau = []

for i in range(1, len(lines)):
    data = lines[i].split()
    chi_d_L.append(float(data[0]))
    chi_CMB.append(float(data[1]))
    N.append(float(data[2]))
    n_s.append(float(data[3]))
    z.append(float(data[4]))
    prop_baryon.append(float(data[5]))
    sigma_temp.append(float(data[6]))
    A.append(float(data[7]))
    b.append(float(data[8]))
    z_0.append(float(data[9]))
    tau.append(float(data[10]))
    
# Creates a list where each element is a solution on the pareto front
    
pareto_front = []

for i in range(len(chi_d_L)):
    pareto_front.append([chi_d_L[i],
                       chi_CMB[i],
                       N[i],
                       n_s[i],
                       z[i],
                       prop_baryon[i],
                       sigma_temp[i],
                       A[i],
                       b[i],
                       z_0[i],
                       tau[i]
                       ])
  
# Sorts the solutions by the CMB fitness (Just preference, can be done with d_L)
# Should not be done with parameter elements, as that mixes solutions together
    
pareto_front.sort(key=lambda x: x[1])
        
# Outputs the solutions for the ordered pareto front as a file
    
file = open("pareto_solutions", "w")
file.write("d_L, CMB, N, n_s, proportion_baryon, sigma_temperature, A, b, z_0, tau")
file.write("\n")
for i in range(len(pareto_front)):
    file.write(f"{pareto_front[i][0]} {pareto_front[i][1]} {pareto_front[i][2]} {pareto_front[i][3]} {pareto_front[i][4]} {pareto_front[i][5]} {pareto_front[i][6]} {pareto_front[i][7]} {pareto_front[i][8]} {pareto_front[i][9]} {pareto_front[i][10]}")
    file.write("\n")
file.close()

# Re-extracts the ordered values for the Pareto front

d_L_list = []
CMB_list = []
N_list = []
n_s_list = []
z_list = []
prop_baryon_list = []
sigma_temp_list = []
A_list = []
b_list = []
z_0_list = []
tau_list = []

for i in range(len(pareto_front)):
    d_L_list.append(float(pareto_front[i][0]))
    CMB_list.append(float(pareto_front[i][1]))
    N_list.append(float(pareto_front[i][2]))
    n_s_list.append(float(pareto_front[i][3]))
    z_list.append(float(pareto_front[i][4]))
    prop_baryon_list.append(float(pareto_front[i][5]))
    sigma_temp_list.append(float(pareto_front[i][6]))
    A_list.append(float(pareto_front[i][7]))
    b_list.append(float(pareto_front[i][8]))
    z_0_list.append(float(pareto_front[i][9]))
    tau_list.append(float(pareto_front[i][10]))
  
# Selects specific solutions to discuss in the paper. We determine this 
# after looking at the plots for all solutions in the pareto front

#chosen_solutions = [36, 115, 180] 

#chosen_solutions = [0, 42, 185] 

chosen_solutions = [0, 110, 260] 

# Plots the pareto front with the chosen solutions given colors

py.scatter(d_L_list, CMB_list, color = "black", label = "Pareto Front")
py.scatter(d_L_list[chosen_solutions[0]], CMB_list[chosen_solutions[0]], color = "red", marker = "h", label = "CMB Solution")
py.scatter(d_L_list[chosen_solutions[1]], CMB_list[chosen_solutions[1]], color = "gold", marker = "s", label = "Preferred Solution")
py.scatter(d_L_list[chosen_solutions[2]], CMB_list[chosen_solutions[2]], color = "cyan", marker = "p", label = r"$d_L$ Solution")
py.ylabel(r"$\text{CMB Fitness}$", fontsize = 15, )
py.xlabel(r"$d_L \text{ Fitness}$", fontsize = 15)
py.legend(loc = "upper right", fontsize = 15)
py.ylim(-1000, 10000)
#py.xlim(111.7, 112.8)
#py.xticks([111.75, 112, 112.25, 112.5, 112.75], size = 15)
py.xlim(111.45, 114.05)
py.xticks([111.5, 112, 112.5, 113, 113.5, 114], size = 15)
#py.xlim(111, 115)
#py.xticks([111, 112, 113, 114, 115], size = 15)
py.yticks(size = 15)
py.savefig('Pareto_Front.pdf', bbox_inches='tight')
py.show()

# Gets the plots for the chosen solutions

for i in range(len(chosen_solutions)):
    
    # Gets the observational data for the Hubble tension plots

    z_L,d_L,e_L,z_L_C,d_L_C,e_L_C,z_L_K,d_L_K,e_L_K,z_L_R,d_L_R,e_L_R,z_L_Sc,d_L_Sc,e_L_Sc,z_L_Su,d_L_Su,e_L_Su,z_L_CMB,d_L_CMB,e_L_CMB = GA.read_data()
    
    # Gets the free parameters of a solution on the front
    
    N = 10**(float(N_list[chosen_solutions[i]]))
    n_s = 10**(float(n_s_list[chosen_solutions[i]]))
    z = 10**(float(z_list[chosen_solutions[i]]))
    proportion_baryon = 10**(float(prop_baryon_list[chosen_solutions[i]]))
    sigma_temperature = 10**(float(sigma_temp_list[chosen_solutions[i]])) 
    A = 10**(float(A_list[chosen_solutions[i]]))
    b = 10**(float(b_list[chosen_solutions[i]]))
    z_0 = 10**(float(z_0_list[chosen_solutions[i]]))
    tau = 10**(float(tau_list[chosen_solutions[i]]))
    
    # Prints out those values for monitoring purposes
    
    print(f"{chosen_solutions[i]} {d_L_list[chosen_solutions[i]]} {CMB_list[chosen_solutions[i]]} {N} {n_s} {z} {proportion_baryon} {sigma_temperature} {A} {b} {z_0} {tau}")
    print()
    
    # Calculates the first order GR-SI power spectrum for this specific
    # solution
    
    GRSI_power_spectrum_values = GA.GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    # Adds the second order corrections to the GR-SI power spectrum
    
    for j in range(len(GRSI_power_spectrum_values)):
        GRSI_power_spectrum_values[j] += C_L[j]
    
    # Calculates the percent error between the GR-SI values and observed values
    
    chi_LCDM_CMB, chi_LCDM_SN, chi_SI, r_CMB, r_SN, r_SI = GA.chi2_3(A, b, z_0, tau)
    
    
    GRSI_residual = []
    GRSI_error = []
    
    # Different ways to compare modeled vs observed data. We use the ratio
    # in the paper
    
    '''
    # Percent Error
    for j in range(len(planck_data)):
        GRSI_res = (GRSI_power_spectrum_values[j] - planck_data[j]) / planck_data[j] * 100
        GRSI_err = (planck_error[j]) / planck_data[j] * 100
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    '''
    # Residual
    for j in range(len(planck_data)):
        GRSI_res = GRSI_power_spectrum_values[j] - planck_data[j]
        GRSI_err = planck_error[j]
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    '''
    # r plot
    for j in range(len(planck_data)):
        GRSI_res = ((GRSI_power_spectrum_values[j] - planck_data[j]) / planck_error[j])**2
        GRSI_err = 0
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    
    # Ratio
    for j in range(len(planck_data)):
        GRSI_res = GRSI_power_spectrum_values[j] / planck_data[j]
        GRSI_err = planck_error[j] / planck_data[j]
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    
    # Creates the combined plot of the power spectrum and its percent error as 
    # seen in the paper
    
    # Interpolates the Second order corrections over many l
    
    points = 1000
    
    CL_l = np.array(l_values)
    CL_CL = np.array(C_L)
    smooth_l = np.linspace(CL_l.min(), CL_l.max(), points)
    spl = make_interp_spline(CL_l, CL_CL, k=1)
    smooth_CL = spl(smooth_l)
    
    smooth_GRSI = []
    
    for k in range(len(smooth_l)):
        smooth_GRSI.append(GA.Individual_GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau, smooth_l[k]) + smooth_CL[k])
    
    # Plots the power spectrum plots seen in the paper
    
    fig = py.figure(figsize=(6.4, 9.6))
    gs = fig.add_gridspec(2, hspace = 0)
    axs = gs.subplots(sharex=True)

    axs[0].errorbar(l_values, planck_data, planck_error, fmt = "s", color = "Black", capsize = 3, markersize = 3, label = r"$\text{Planck 2018 Data}$", zorder = 1)
    axs[0].errorbar(smooth_l, smooth_GRSI, color = "Red", label = r"$\text{GR-SI Model}$", zorder = 1)
    axs[0].set_ylabel(r"$l(l + 1) \text{ } C_{TT} \text{ } / \text{ } 2π$", size = 18)
    axs[0].legend(loc = "upper right", fontsize = 13)
    y_line = [1000, 2000, 3000, 4000, 5000, 6000]
    axs[0].set_yticks(y_line)
    # Percent error
    #axs[1].errorbar(l_values, GRSI_residual, GRSI_error, color = "Red", label = "GRSI", fmt = "o", capsize = 2, markersize = 3)
    #axs[1].plot([0, max(l_values) + 30], [0,0], color ="Black", label = r"$\text{Planck 2018}$")
    #axs[1].set_ylabel(r"$\text{Percent Error}$", size = 18)
    #y_line = [-4, -2, 0, 2, 4]
    #axs[1].set_yticks(y_line)
    #axs[1].set_ylim(-5,5)
    # Residual
    #axs[1].errorbar(l_values, GRSI_residual, GRSI_error, color = "Red", label = "GRSI", fmt = "o", capsize = 2, markersize = 3)
    #axs[1].plot([0, max(l_values) + 30], [0,0], color ="Black", label = r"$\text{Planck 2018}$")
    #axs[1].set_ylabel(r"$\text{Residual}$", size = 18)
    #axs[1].set_ylim(-150,150)
    # r
    #axs[1].scatter(l_values, GRSI_residual, color = "Red", label = "GRSI")
    #axs[1].set_ylabel(r"$\text{r Value}$", size = 18)
    #axs[1].set_ylim([1e-4,1.5e5])
    #axs[1].set_yscale('log')
    # GRSI / Planck ratio
    axs[1].scatter(l_values, GRSI_residual, color = "Red", label = "GR-SI")
    axs[1].plot([0, max(l_values) + 30], [1.01,1.01], color ="Black", label = "1 Percent Error", alpha = 0.5)
    axs[1].plot([0, max(l_values) + 30], [0.99,0.99], color ="Black", alpha = 0.5)
    axs[1].set_ylabel(r"$\text{Ratio}$", size = 18)
    axs[1].set_ylim([0.95,1.05])
    x_line = [0, 250, 500, 750, 1000, 1250, 1500, 1750]
    py.xticks(x_line, size = 15)
    py.xlabel(r"$l \text{ moment}$", size = 18)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[1].legend(loc = "upper right", fontsize = 13)
    py.text(0, 1.055, "(a)")
    py.text(0, 1.0425, "(b)")
    if i == 0:
        py.savefig('Low_CMB_CMB.pdf', bbox_inches='tight')
    if i == 1:
        py.savefig('Selected_CMB.pdf', bbox_inches='tight')
    if i == 2:
        py.savefig('Low_HT_CMB.pdf', bbox_inches='tight')
    py.show()
    
    # Sets up the information for the luminosity distance plots
    
    length = 400
    z_min_LCDM = 1e-4
    z_max_LCDM = 1089.0
    l_zv = np.mgrid[math.log10(z_min_LCDM):math.log10(z_max_LCDM):length*1j]
    z_CDM = []
    z_SI = []
    D_L_CDM_1 = []
    D_L_CDM_2 = []
    D_L_22 = []
    D_L_22_min = []
    D_L_22_max = []
    D_M_22 = []
    D_M_22_min = []
    D_M_22_max = []
    Npts = 3
    
    # Gets the modeled luminosity distances for high-z Lambda CDM, low-z Lambda
    # CDM, and GR-SI
    
    for j in range(0,length):
        
        # Gets the z values
        
        z = 10**l_zv[j]
    
        # High-z Lambda CDM luminosity distance calculations
        arg_CDM = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_1,O_M_1,O_R_1,O_L_1))[0]
        D_L_CDM_1.append(((1+z)/(H_0_1*math.sqrt(O_K_1)))*math.sinh(math.sqrt(O_K_1)*arg_CDM))
        
        
        # Low-z Lambda CDM luminosity distance calculations
        arg_CDM = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_2,O_M_2,O_R_2,O_L_2))[0]
        D_L_CDM_2.append(((1+z)/(H_0_2*math.sqrt(O_K_2)))*math.sinh(math.sqrt(O_K_2)*arg_CDM))
        z_CDM.append(z)
    
        # GR-SI luminosity distance calculations 
        arg = integrate.quad(GA.SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A,b,z_0,tau))[0]
        try: 
            D_Mv = 1 - 1 / (1 + math.exp((z - z_0) / tau)) + A * math.exp(-z / b)
        except OverflowError:
            D_Mv = 1 + A * math.exp(-z / b)
        
        D_M_22.append(D_Mv)
        D_L_22.append(((1+z)/(H_0_GRSI))*arg)
    
        
    xv_1 = [1089,1089]
    yv_1 = [1e-1,1e6]
    
    lw0=1.0
    lw1=0.25
    
    tag_1b = r'$\text{GR-SI}: \ h=%5.2f$' % (H_0_GRSI / 100)

    tag_2b = r'$\Lambda{\rm CDM} \text{ (CMB)}: \ h=0.67$'

    tag_3b = r'$\Lambda{\rm CDM} \text{ (SN)}: \ h=0.73$'
    
    ms0=3
    
    # Defines res_SN, res_CMB, res_GRSI
    
    res_SN = []
    res_CMB = []
    res_GRSI = []
    d_L_error = []
    
    for k in range(len(z_L)):
        
        # High-z Lambda CDM
        z = z_L[k]
        arg_CDM_1 = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_1,O_M_1,O_R_1,O_L_1))[0]
        d_L_CDM_1 = (((1+z)/(H_0_1*math.sqrt(O_K_1)))*math.sinh(math.sqrt(O_K_1)*arg_CDM_1))
        
        # Low-z Lambda CDM
        arg_CDM_2 = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_2,O_M_2,O_R_2,O_L_2))[0]
        d_L_CDM_2 = (((1+z)/(H_0_2*math.sqrt(O_K_2)))*math.sinh(math.sqrt(O_K_2)*arg_CDM_2))
        
        # 100 h for H_0 in km / Mpc / s
        
        # GR-SI
        arg_SI = integrate.quad(GA.SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A, b, z_0, tau))[0]
        d_L_SI = (((1+z)/(H_0_GRSI))*arg_SI)
        
        # r value is r_SN, r_CMB, r_SI implicitly
        
        # Residual
        '''
        res_SN_value = (d_L_CDM_2 - d_L[k])
        res_SN.append(res_SN_value)
        res_CMB_value = d_L_CDM_1 - d_L[k]
        res_CMB.append(res_CMB_value)
        res_GRSI_value = d_L_SI - d_L[k]
        res_GRSI.append(res_GRSI_value)
        d_L_error.append(e_L[k])
        '''
        '''
        # Percent Error
        res_SN_value = (d_L_CDM_2 - d_L[k]) / d_L[k] * 100
        res_SN.append(res_SN_value)
        res_CMB_value = (d_L_CDM_1 - d_L[k]) / d_L[k] * 100
        res_CMB.append(res_CMB_value)
        res_GRSI_value = (d_L_SI - d_L[k]) / d_L[k] * 100
        res_GRSI.append(res_GRSI_value)
        error_d_L = e_L[k] / d_L[k] * 100
        d_L_error.append(error_d_L)
        
        '''
        # Ratio
        res_SN_value = d_L_CDM_2 / d_L[k]
        res_SN.append(res_SN_value)
        res_CMB_value = d_L_CDM_1 / d_L[k]
        res_CMB.append(res_CMB_value)
        res_GRSI_value = d_L_SI / d_L[k]
        res_GRSI.append(res_GRSI_value)
        error_d_L = e_L[k] / d_L[k]
        d_L_error.append(error_d_L)
        
    # Shifts the points for visual clarity (lowercase to avoid conflicting name)
        
    z_L_grsi = []
    z_L_sn = []
    z_L_cmb = []
    
    for p in range(len(z_L)):
        z_L_grsi.append(z_L[p])
        z_L_sn.append(z_L[p] + z_L[p] *.1)
        z_L_cmb.append(z_L[p] - z_L[p] *.1)
           
    # Combines the luminosity distance plot with the r plot as seen in paper
    
    # Also puts all of the observational data together
    
    # Scales the d_L to Mpc
    
    speed_of_light = 3*10**5 #(km/s) 
        
    for j in range(len(D_L_22)):
        D_L_22[j] = D_L_22[j] * speed_of_light
            
    for j in range(len(D_L_CDM_1)):
        D_L_CDM_1[j] = D_L_CDM_1[j] * speed_of_light
        
    for j in range(len(D_L_CDM_2)):
        D_L_CDM_2[j] = D_L_CDM_2[j] * speed_of_light
        
    for j in range(len(d_L)):
        d_L[j] = d_L[j] * speed_of_light
        
    for j in range(len(e_L)):
        e_L[j] = e_L[j] * speed_of_light
        
    # Plots the d_L plots in the paper          
    
    fig = py.figure(figsize=(6.4, 9.6))
    gs = fig.add_gridspec(2, hspace = 0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(z_CDM,D_L_22,color='r',lw=lw0,label=tag_1b, zorder = 2)
    axs[0].plot(z_CDM,D_L_CDM_1,color='g',linestyle='dashed',lw=lw0,label=tag_2b, zorder = 2)
    axs[0].plot(z_CDM,D_L_CDM_2,color='b',linestyle='dotted',lw=lw0,label=tag_3b, zorder = 2)
    axs[0].errorbar(z_L,d_L,yerr=e_L, marker='s', mfc='black', mec='black', ecolor='black', ms=ms0, ls='None', label=r'Observational Data', zorder = 1)
    axs[0].set_ylim([1,1e10])
    axs[0].set_ylabel((r"$ d_L \text{ (Mpc)}$"), size = 18)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(loc='upper left',shadow=False,markerscale=2,scatterpoints=1,numpoints=1,frameon=False,fontsize=13)
    # Residual
    #axs[1].set_ylabel((r"$\text{Residual}$"), size = 18)
    #axs[1].set_ylim([-0.13,0.13])
    #axs[1].errorbar(z_L,res_GRSI,d_L_error,marker = 'o',capsize = 2,mfc='red', mec='red', ecolor = 'red',ms=ms0, ls='None', label = tag_1b)
    #axs[1].errorbar(z_L,res_CMB,d_L_error,marker = 'o',capsize = 2,mfc='green', mec='green',ecolor='green',ms=ms0, ls='None', label=tag_2b)
    #axs[1].errorbar(z_L,res_SN,d_L_error,marker = 'o',capsize = 2,mfc='blue', mec='blue',ecolor='blue',ms=ms0, ls='None', label=tag_3b)
    #axs[1].legend(loc='upper left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # r plot
    #axs[1].set_ylabel((r"$\text{r Value}$"), size = 18)
    #axs[1].set_ylim([1e-4,1.5e5])
    #axs[1].set_yscale('log')
    #axs[1].scatter(z_L,r_SI,c='r',s=s0,edgecolors='r',label=tag_1b)
    #axs[1].scatter(z_L,r_CMB,c='g',s=s0,edgecolors='g',label=tag_2b)
    #axs[1].scatter(z_L,r_SN,c='b',s=s0,edgecolors='b',label=tag_3b)
    #axs[1].legend(loc='upper left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # Percent Error
    #axs[1].set_ylabel((r"$\text{Percent Error}$"), size = 18)
    #axs[1].set_ylim([-40,40])
    #axs[1].scatter(z_L,res_GRSI, color = 'red',s = 12, label = tag_1b)
    #axs[1].scatter(z_L,res_CMB,color = 'green',s = 12, label=tag_2b)
    #axs[1].scatter(z_L,res_SN,color = 'blue',s = 12, label=tag_3b)
    #axs[1].legend(loc='lower left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # Ratio
    axs[1].set_ylabel((r"$\text{Ratio}$"), size = 18)
    axs[1].set_ylim([0.5,1.5])
    axs[1].scatter(z_L_grsi,res_GRSI, color = 'red',s = 12, label = tag_1b)
    axs[1].scatter(z_L_cmb,res_CMB,color = 'green', marker = "s",s = 12, label=tag_2b)
    axs[1].scatter(z_L_sn,res_SN,color = 'blue', marker = "^",s = 12, label=tag_3b)
    axs[1].legend(loc='lower left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    axs[1].set_xlim([1e-2,1500])
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    py.text(1.8*10**(-2), 1.55, "(c)")
    py.text(1.8*10**(-2), 1.425, "(d)")
    py.xlabel(r"$z$", size = 18)
    if i == 0:
        py.savefig('Low_CMB_HT.pdf', dpi=600, bbox_inches='tight')
    if i == 1:
        py.savefig('Selected_HT.pdf', dpi=600, bbox_inches='tight')
    if i == 2:
        py.savefig('Low_HT_HT.pdf', dpi=600, bbox_inches='tight')
    py.show()
   
#
#
#
# Note: Don't run both at once, comment out the undesired block of code
#
#
#
"""
# Gets the plots for each solution on the meta front.

for i in range(len(d_L_list)):
    
    N = 10**(float(N_list[i]))
    n_s = 10**(float(n_s_list[i]))
    z = 10**(float(z_list[i]))
    proportion_baryon = 10**(float(prop_baryon_list[i]))
    sigma_temperature = 10**(float(sigma_temp_list[i])) 
    A = 10**(float(A_list[i]))
    b = 10**(float(b_list[i]))
    z_0 = 10**(float(z_0_list[i]))
    tau = 10**(float(tau_list[i]))

    print(f"{i} {d_L_list[i]} {CMB_list[i]} {N} {n_s} {z} {proportion_baryon} {sigma_temperature} {A} {b} {z_0} {tau}")
    print()
    
    # Gets the observational data for the Hubble tension plots

    z_L,d_L,e_L,z_L_C,d_L_C,e_L_C,z_L_K,d_L_K,e_L_K,z_L_R,d_L_R,e_L_R,z_L_Sc,d_L_Sc,e_L_Sc,z_L_Su,d_L_Su,e_L_Su,z_L_CMB,d_L_CMB,e_L_CMB = GA.read_data()
    
    # Gets the free parameters of a solution on the front
    
    GRSI_power_spectrum_values = GA.GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau)
    
    # Adds the second order corrections to the GR-SI power spectrum
    
    for j in range(len(GRSI_power_spectrum_values)):
        GRSI_power_spectrum_values[j] += C_L[j]
    
    # Calculates the percent error between the GR-SI values and observed values
    
    chi_LCDM_CMB, chi_LCDM_SN, chi_SI, r_CMB, r_SN, r_SI = GA.chi2_3(A, b, z_0, tau)
    
    GRSI_residual = []
    GRSI_error = []
    
    # Different ways to compare modeled vs observed data. We use the ratio
    # in the paper
    
    '''
    # Percent Error
    for j in range(len(planck_data)):
        GRSI_res = (GRSI_power_spectrum_values[j] - planck_data[j]) / planck_data[j] * 100
        GRSI_err = (planck_error[j]) / planck_data[j] * 100
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    '''
    # Residual
    for j in range(len(planck_data)):
        GRSI_res = GRSI_power_spectrum_values[j] - planck_data[j]
        GRSI_err = planck_error[j]
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    '''
    # r plot
    for j in range(len(planck_data)):
        GRSI_res = ((GRSI_power_spectrum_values[j] - planck_data[j]) / planck_error[j])**2
        GRSI_err = 0
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    '''
    
    # Ratio
    for j in range(len(planck_data)):
        GRSI_res = GRSI_power_spectrum_values[j] / planck_data[j]
        GRSI_err = planck_error[j] / planck_data[j]
        GRSI_error.append(GRSI_err)
        GRSI_residual.append(GRSI_res)
    
    # Creates the combined plot of the power spectrum and its percent error as 
    # seen in the paper
    
    # Interpolates the Second order corrections over many l
    
    points = 1000
    
    CL_l = np.array(l_values)
    CL_CL = np.array(C_L)
    smooth_l = np.linspace(CL_l.min(), CL_l.max(), points)
    spl = make_interp_spline(CL_l, CL_CL, k=1)
    smooth_CL = spl(smooth_l)
    
    smooth_GRSI = []
    
    for k in range(len(smooth_l)):
        smooth_GRSI.append(GA.Individual_GRSI_power_spectrum(N, n_s, z, proportion_baryon, sigma_temperature, A, b, z_0, tau, smooth_l[k]) + smooth_CL[k])
        
    # Plots the power spectrum plots seen in the paper
    
    fig = py.figure(figsize=(6.4, 9.6))
    gs = fig.add_gridspec(2, hspace = 0)
    axs = gs.subplots(sharex=True)

    axs[0].errorbar(l_values, planck_data, planck_error, fmt = "s", color = "Black", capsize = 3, markersize = 3, label = r"$\text{Planck 2018 Data}$", zorder = 1)
    axs[0].errorbar(smooth_l, smooth_GRSI, color = "Red", label = r"$\text{GR-SI Model}$", zorder = 1)
    axs[0].set_ylabel(r"$l(l + 1) \text{ } C_{TT} \text{ } / \text{ } 2π$", size = 18)
    axs[0].legend(loc = "upper right", fontsize = 13)
    y_line = [1000, 2000, 3000, 4000, 5000, 6000]
    axs[0].set_yticks(y_line)
    # Percent error
    #axs[1].errorbar(l_values, GRSI_residual, GRSI_error, color = "Red", label = "GRSI", fmt = "o", capsize = 2, markersize = 3)
    #axs[1].plot([0, max(l_values) + 30], [0,0], color ="Black", label = r"$\text{Planck 2018}$")
    #axs[1].set_ylabel(r"$\text{Percent Error}$", size = 18)
    #y_line = [-4, -2, 0, 2, 4]
    #axs[1].set_yticks(y_line)
    #axs[1].set_ylim(-5,5)
    # Residual
    #axs[1].errorbar(l_values, GRSI_residual, GRSI_error, color = "Red", label = "GRSI", fmt = "o", capsize = 2, markersize = 3)
    #axs[1].plot([0, max(l_values) + 30], [0,0], color ="Black", label = r"$\text{Planck 2018}$")
    #axs[1].set_ylabel(r"$\text{Residual}$", size = 18)
    #axs[1].set_ylim(-150,150)
    # r
    #axs[1].scatter(l_values, GRSI_residual, color = "Red", label = "GRSI")
    #axs[1].set_ylabel(r"$\text{r Value}$", size = 18)
    #axs[1].set_ylim([1e-4,1.5e5])
    #axs[1].set_yscale('log')
    # GRSI / Planck ratio
    axs[1].scatter(l_values, GRSI_residual, color = "Red", label = "GR-SI")
    axs[1].plot([0, max(l_values) + 30], [1.01,1.01], color ="Black", label = "1 Percent Error", alpha = 0.5)
    axs[1].plot([0, max(l_values) + 30], [0.99,0.99], color ="Black", alpha = 0.5)
    axs[1].set_ylabel(r"$\text{Ratio}$", size = 18)
    axs[1].set_ylim([0.95,1.05])
    x_line = [0, 250, 500, 750, 1000, 1250, 1500, 1750]
    py.xticks(x_line, size = 15)
    py.xlabel(r"$l \text{ moment}$", size = 18)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[1].legend(loc = "upper right", fontsize = 13)
    py.text(0, 1.055, "(a)")
    py.text(0, 1.0425, "(b)")
    if i == 0:
        py.savefig('Low_CMB_CMB.pdf', bbox_inches='tight')
    if i == 1:
        py.savefig('Selected_CMB.pdf', bbox_inches='tight')
    if i == 2:
        py.savefig('Low_HT_CMB.pdf', bbox_inches='tight')
    py.show()
    
    # Sets up the information for the luminosity distance plots
    
    length = 400
    z_min_LCDM = 1e-4
    z_max_LCDM = 1089.0
    l_zv = np.mgrid[math.log10(z_min_LCDM):math.log10(z_max_LCDM):length*1j]
    z_CDM = []
    z_SI = []
    D_L_CDM_1 = []
    D_L_CDM_2 = []
    D_L_22 = []
    D_L_22_min = []
    D_L_22_max = []
    D_M_22 = []
    D_M_22_min = []
    D_M_22_max = []
    Npts = 3
    
    # Gets the modeled luminosity distances for high-z Lambda CDM, low-z Lambda
    # CDM, and GR-SI
    
    for j in range(0,length):
        
        # Gets the z values
        
        z = 10**l_zv[j]
    
        # High-z Lambda CDM luminosity distance calculations
        arg_CDM = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_1,O_M_1,O_R_1,O_L_1))[0]
        D_L_CDM_1.append(((1+z)/(H_0_1*math.sqrt(O_K_1)))*math.sinh(math.sqrt(O_K_1)*arg_CDM))
        
        
        # Low-z Lambda CDM luminosity distance calculations
        arg_CDM = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_2,O_M_2,O_R_2,O_L_2))[0]
        D_L_CDM_2.append(((1+z)/(H_0_2*math.sqrt(O_K_2)))*math.sinh(math.sqrt(O_K_2)*arg_CDM))
        z_CDM.append(z)
    
        # GR-SI luminosity distance calculations 
        arg = integrate.quad(GA.SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A,b,z_0,tau))[0]
        try: 
            D_Mv = 1 - 1 / (1 + math.exp((z - z_0) / tau)) + A * math.exp(-z / b)
        except OverflowError:
            D_Mv = 1 + A * math.exp(-z / b)
        
        D_M_22.append(D_Mv)
        D_L_22.append(((1+z)/(H_0_GRSI))*arg)
    
        
    xv_1 = [1089,1089]
    yv_1 = [1e-1,1e6]
    
    lw0=1.0
    lw1=0.25
    
    tag_1b = r'$\text{GR-SI}: \ h=%5.2f$' % (H_0_GRSI / 100)

    tag_2b = r'$\Lambda{\rm CDM} \text{ (CMB)}: \ h=0.67$'

    tag_3b = r'$\Lambda{\rm CDM} \text{ (SN)}: \ h=0.73$'
    
    ms0=3
    
    # Defines res_SN, res_CMB, res_GRSI
    
    res_SN = []
    res_CMB = []
    res_GRSI = []
    d_L_error = []
    
    for k in range(len(z_L)):
        
        # High-z Lambda CDM
        z = z_L[k]
        arg_CDM_1 = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_1,O_M_1,O_R_1,O_L_1))[0]
        d_L_CDM_1 = (((1+z)/(H_0_1*math.sqrt(O_K_1)))*math.sinh(math.sqrt(O_K_1)*arg_CDM_1))
        
        # Low-z Lambda CDM
        arg_CDM_2 = integrate.quad(GA.CDM_int, 1/(1+z), 1, args=(O_K_2,O_M_2,O_R_2,O_L_2))[0]
        d_L_CDM_2 = (((1+z)/(H_0_2*math.sqrt(O_K_2)))*math.sinh(math.sqrt(O_K_2)*arg_CDM_2))
        
        # 100 h for H_0 in km / Mpc / s
        
        # GR-SI
        arg_SI = integrate.quad(GA.SI_int, 1/(1+z), 1, args=(Omega_M,Omega_R,Omega_L,A, b, z_0, tau))[0]
        d_L_SI = (((1+z)/(H_0_GRSI))*arg_SI)
        
        # r value is r_SN, r_CMB, r_SI implicitly
        
        # Residual
        '''
        res_SN_value = (d_L_CDM_2 - d_L[k])
        res_SN.append(res_SN_value)
        res_CMB_value = d_L_CDM_1 - d_L[k]
        res_CMB.append(res_CMB_value)
        res_GRSI_value = d_L_SI - d_L[k]
        res_GRSI.append(res_GRSI_value)
        d_L_error.append(e_L[k])
        '''
        '''
        # Percent Error
        res_SN_value = (d_L_CDM_2 - d_L[k]) / d_L[k] * 100
        res_SN.append(res_SN_value)
        res_CMB_value = (d_L_CDM_1 - d_L[k]) / d_L[k] * 100
        res_CMB.append(res_CMB_value)
        res_GRSI_value = (d_L_SI - d_L[k]) / d_L[k] * 100
        res_GRSI.append(res_GRSI_value)
        error_d_L = e_L[k] / d_L[k] * 100
        d_L_error.append(error_d_L)
        
        '''
        # Ratio
        res_SN_value = d_L_CDM_2 / d_L[k]
        res_SN.append(res_SN_value)
        res_CMB_value = d_L_CDM_1 / d_L[k]
        res_CMB.append(res_CMB_value)
        res_GRSI_value = d_L_SI / d_L[k]
        res_GRSI.append(res_GRSI_value)
        error_d_L = e_L[k] / d_L[k]
        d_L_error.append(error_d_L)
        
    # Shifts the points for visual clarity (lowercase to avoid conflicting name)
        
    z_L_grsi = []
    z_L_sn = []
    z_L_cmb = []
    
    for p in range(len(z_L)):
        z_L_grsi.append(z_L[p])
        z_L_sn.append(z_L[p] + z_L[p] *.1)
        z_L_cmb.append(z_L[p] - z_L[p] *.1)
           
    # Combines the luminosity distance plot with the r plot as seen in paper
    
    # Also puts all of the observational data together
    
    # Scales the d_L to Mpc
    
    speed_of_light = 3*10**5 #(km/s) 
        
    for j in range(len(D_L_22)):
        D_L_22[j] = D_L_22[j] * speed_of_light
            
    for j in range(len(D_L_CDM_1)):
        D_L_CDM_1[j] = D_L_CDM_1[j] * speed_of_light
        
    for j in range(len(D_L_CDM_2)):
        D_L_CDM_2[j] = D_L_CDM_2[j] * speed_of_light
        
    for j in range(len(d_L)):
        d_L[j] = d_L[j] * speed_of_light
        
    for j in range(len(e_L)):
        e_L[j] = e_L[j] * speed_of_light
        
    # Plots the d_L plots in the paper          
    
    fig = py.figure(figsize=(6.4, 9.6))
    gs = fig.add_gridspec(2, hspace = 0)
    axs = gs.subplots(sharex=True)
    axs[0].plot(z_CDM,D_L_22,color='r',lw=lw0,label=tag_1b, zorder = 2)
    axs[0].plot(z_CDM,D_L_CDM_1,color='g',linestyle='dashed',lw=lw0,label=tag_2b, zorder = 2)
    axs[0].plot(z_CDM,D_L_CDM_2,color='b',linestyle='dotted',lw=lw0,label=tag_3b, zorder = 2)
    axs[0].errorbar(z_L,d_L,yerr=e_L, marker='s', mfc='black', mec='black', ecolor='black', ms=ms0, ls='None', label=r'Observational Data', zorder = 1)
    axs[0].set_ylim([1,1e10])
    axs[0].set_ylabel((r"$ d_L \text{ (Mpc)}$"), size = 18)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].legend(loc='upper left',shadow=False,markerscale=2,scatterpoints=1,numpoints=1,frameon=False,fontsize=13)
    # Residual
    #axs[1].set_ylabel((r"$\text{Residual}$"), size = 18)
    #axs[1].set_ylim([-0.13,0.13])
    #axs[1].errorbar(z_L,res_GRSI,d_L_error,marker = 'o',capsize = 2,mfc='red', mec='red', ecolor = 'red',ms=ms0, ls='None', label = tag_1b)
    #axs[1].errorbar(z_L,res_CMB,d_L_error,marker = 'o',capsize = 2,mfc='green', mec='green',ecolor='green',ms=ms0, ls='None', label=tag_2b)
    #axs[1].errorbar(z_L,res_SN,d_L_error,marker = 'o',capsize = 2,mfc='blue', mec='blue',ecolor='blue',ms=ms0, ls='None', label=tag_3b)
    #axs[1].legend(loc='upper left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # r plot
    #axs[1].set_ylabel((r"$\text{r Value}$"), size = 18)
    #axs[1].set_ylim([1e-4,1.5e5])
    #axs[1].set_yscale('log')
    #axs[1].scatter(z_L,r_SI,c='r',s=s0,edgecolors='r',label=tag_1b)
    #axs[1].scatter(z_L,r_CMB,c='g',s=s0,edgecolors='g',label=tag_2b)
    #axs[1].scatter(z_L,r_SN,c='b',s=s0,edgecolors='b',label=tag_3b)
    #axs[1].legend(loc='upper left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # Percent Error
    #axs[1].set_ylabel((r"$\text{Percent Error}$"), size = 18)
    #axs[1].set_ylim([-40,40])
    #axs[1].scatter(z_L,res_GRSI, color = 'red',s = 12, label = tag_1b)
    #axs[1].scatter(z_L,res_CMB,color = 'green',s = 12, label=tag_2b)
    #axs[1].scatter(z_L,res_SN,color = 'blue',s = 12, label=tag_3b)
    #axs[1].legend(loc='lower left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    # Ratio
    axs[1].set_ylabel((r"$\text{Ratio}$"), size = 18)
    axs[1].set_ylim([0.5,1.5])
    axs[1].scatter(z_L_grsi,res_GRSI, color = 'red',s = 12, label = tag_1b)
    axs[1].scatter(z_L_cmb,res_CMB,color = 'green', marker = "s",s = 12, label=tag_2b)
    axs[1].scatter(z_L_sn,res_SN,color = 'blue', marker = "^",s = 12, label=tag_3b)
    axs[1].legend(loc='lower left',shadow=False,markerscale=1,scatterpoints=1,frameon=False,fontsize=13)
    axs[1].set_xlim([1e-2,1500])
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    py.text(1.8*10**(-2), 1.55, "(c)")
    py.text(1.8*10**(-2), 1.425, "(d)")
    py.xlabel(r"$z$", size = 18)
    if i == 0:
        py.savefig('Low_CMB_HT.pdf', dpi=600, bbox_inches='tight')
    if i == 1:
        py.savefig('Selected_HT.pdf', dpi=600, bbox_inches='tight')
    if i == 2:
        py.savefig('Low_HT_HT.pdf', dpi=600, bbox_inches='tight')
    py.show()
"""


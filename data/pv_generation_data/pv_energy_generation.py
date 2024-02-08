import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import pvsystem
from datetime import datetime
from parameters_pv import parameters
from mpl_toolkits.mplot3d import Axes3D

series_panel = 5
parallel_panel = 3 
#CFPV Data
PCE_ref_CFPV = 10
#y=mx+b
slope_2x_enhance = (-1/100)
constant_2x_enhance = 20
irradiance = np.linspace(0, 1000, 50)  # From 0 to 1 sun (1000 W/m^2)
temperature = np.linspace(0, 35, 50)  # Temperature range


def pv_generation(irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV):
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        irradiance,
        temperature,
        alpha_sc=parameters['alpha_sc'],
        a_ref=parameters['a_ref'],
        I_L_ref=parameters['I_L_ref'],
        I_o_ref=parameters['I_o_ref'],
        R_sh_ref=parameters['R_sh_ref'],
        R_s=parameters['R_s'],
        EgRef=1.121,
        dEgdT=-0.0002677
    )

    # plug the parameters into the SDE and solve for IV curves:
    curve_info = pvsystem.singlediode(
        photocurrent=IL,
        saturation_current=I0,
        resistance_series=Rs,
        resistance_shunt=Rsh,
        nNsVth=nNsVth,
        ivcurve_pnts=100,
        method='lambertw'
    )

    Cell_result = pd.DataFrame({
        'i_sc': curve_info['i_sc'],
        'v_oc': curve_info['v_oc'],
        'i_mp': curve_info['i_mp'],
        'v_mp': curve_info['v_mp'],
        'p_mp': curve_info['p_mp'],
    })
    # .set_index(irradiance.index)

    Total_PV = pd.DataFrame({
        'Irradiance': irradiance,
        'V': Cell_result['v_mp']*series_panel,
        'I': Cell_result['i_mp']*parallel_panel,
    })
    return Total_PV

# irradiance, temperature = get_csv_data(temperature_file, irradiance_file)
Total_PV = pv_generation(irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV)

Total_PV['P'] = Total_PV['I']*Total_PV['V']
#calculating other technology performance
Total_PV['PCE@GHI'] = slope_2x_enhance * Total_PV['Irradiance'] + constant_2x_enhance  #y = mx+b
Total_PV['P_PCE@GHI'] = 20 
Total_PV['P_CFPV'] = Total_PV['P']*(Total_PV['PCE@GHI']/PCE_ref_CFPV) # P = P_silicon * (Enhanced_PCE @ Iradiance level / Silicon PCE efficiency) 
    
print(Total_PV)
plt.plot(Total_PV['Irradiance'])
plt.show()
plt.plot(Total_PV.index, Total_PV[['P','P_CFPV']])
Total_PV[['P','P_CFPV']].to_csv(f'data/pv_generation_data/pv_profiles/profile_year.csv') 
plt.show()


I, T = np.meshgrid(irradiance, temperature)
PV1, PV2 = np.meshgrid(Total_PV['P'], Total_PV['P_CFPV'])
PC1, PC2 = np.meshgrid(Total_PV['P_PCE@GHI'],Total_PV['PCE@GHI'])

# Plotting the enhanced 3D plot with relative PCE visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot for material PV1 with relative PCE as color
surf1 = ax.plot_surface(I, T, PV1, facecolors=plt.cm.viridis(PC1), 
                        rstride=1, cstride=1, alpha=0.8)

# Plot for material PV2 with relative PCE as color
surf2 = ax.plot_surface(I, T, PV2, facecolors=plt.cm.viridis(PC2), 
                        rstride=1, cstride=1, alpha=0.8)

# Labels and title
ax.set_xlabel('Irradiance (W/m^2)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Power Output (kW)')
ax.set_title('Power Output and Relative PCE for Two Different Solar PV Materials')

# Color bar for relative PCE
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), shrink=0.5, aspect=5)
cbar.set_label('Relative PCE')

plt.show()
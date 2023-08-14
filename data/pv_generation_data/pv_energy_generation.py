import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pvlib import pvsystem
from datetime import datetime
from parameters_pv import parameters

series_panel = 5
parallel_panel = 3 
#CFPV Data
PCE_ref_CFPV = 10
#y=mx+b
slope_2x_enhance = (-1/100)
constant_2x_enhance = 20
temperature_file = "temperature.csv"
irradiance_file = "irradiance.csv"

with open(f'{temperature_file}', newline='') as temperature_data, open(f'{irradiance_file}', newline='') as irradiance_data:
    irradiance = pd.read_csv(irradiance_data)
    irradiance['index_date'] = pd.to_datetime(irradiance['index_date'])
    irradiance.set_index(irradiance['index_date'], inplace=True)
  
    temperature = pd.read_csv(temperature_data)
    temperature['valid_time'] = pd.to_datetime(temperature['valid_time'])
    temperature.set_index(temperature['valid_time'], inplace=True)

irradiance = irradiance.loc['2023-01-29']
temperature = temperature.loc['2023-01-29']

def pv_generation(irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV):
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
        irradiance['GHI'],
        temperature['t2m'],
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
    }).set_index(irradiance.index)

    Total_PV = pd.DataFrame({
        'Irradiance': irradiance['GHI'],
        'V': Cell_result['v_mp']*series_panel,
        'I': Cell_result['i_mp']*parallel_panel,
    })
    return Total_PV

Total_PV = pv_generation(irradiance, temperature, series_panel, parallel_panel, PCE_ref_CFPV)

Total_PV['P'] = Total_PV['I']*Total_PV['V']
#calculating other technology performance
Total_PV['PCE@GHI'] = slope_2x_enhance * Total_PV['Irradiance'] + constant_2x_enhance  #y = mx+b
Total_PV['P_CFPV'] = Total_PV['P']*(Total_PV['PCE@GHI']/PCE_ref_CFPV) # P = P_silicon * (Enhanced_PCE @ Iradiance level / Silicon PCE efficiency) 
    
print(Total_PV)
plt.plot(Total_PV['Irradiance'])
plt.show()
plt.plot(Total_PV.index, Total_PV[['P','P_CFPV']])
Total_PV[['P','P_CFPV']].to_csv(f'Profile_June.csv') 
plt.show()

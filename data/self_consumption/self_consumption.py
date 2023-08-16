######## This is how this script will determine both cases ######
# 1. Renewable Energy Source Alone (Without Battery):
# Self-Consumption (%) = (Energy Consumed On-Site / Total Energy Generated) * 100
# 2. Renewable Energy Source with Battery Storage:
# Energy Stored in Battery = Energy Generated - Energy Consumed On-Site (when generation exceeds consumption)
# Energy Discharged from Battery = Energy Consumed On-Site - Energy Generated (when consumption exceeds generation)
# Total Energy Generated = Sum of Energy Consumed On-Site + Energy Stored in Battery (when generation exceeds consumption)
# Self-Consumption (%) = (Energy Consumed On-Site + Energy Discharged from Battery) / Total Energy Generated) * 100

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

self_consumption_file = 'data/plots/self_consumption_january.png'
battery_capacity = 10000
SoC_battery = 40

file_path = glob.glob('data/network_data/network/load_profiles/*.csv')
combined_load_profile = pd.concat((pd.read_csv(f, sep=',') for f in file_path), ignore_index=True)
combined_load_profile = pd.DataFrame(combined_load_profile)
combined_load_profile['mult'] = combined_load_profile['mult']*1000


print(combined_load_profile)

plt.plot(combined_load_profile['mult'])
plt.savefig('data/plots/combined_load_profile.png')
plt.show()

with open('data/network_data/network/Load_profile_1.csv', newline='') as loadprofile_file, open('data/pv_generation_data/pv_profiles/profile_january.csv', newline='') as pvprofile_file:
    pv_profile = pd.read_csv(pvprofile_file)
    load_profile = pd.read_csv(loadprofile_file)
    load_profile['mult'] = load_profile['mult']*1000

#CFPV 
battery_energy = pd.DataFrame(np.zeros(1440), columns=['energy'])
battery_energy['energy'].iloc[0] = battery_capacity*(SoC_battery/100)

for i in range(len(pv_profile)-1):
    p_generated = pv_profile['P'].iloc[i]/60
    cfpv_generated = pv_profile['P_CFPV'].iloc[i]/60
    consumption = load_profile['mult'].iloc[i]/60
    
    if consumption > cfpv_generated:
        energy_demand_battery = (consumption - cfpv_generated)

        if battery_energy['energy'].iloc[i] > energy_demand_battery:
            battery_energy['energy'].iloc[i+1] = battery_energy['energy'].iloc[i] - energy_demand_battery

        else:
            grid_consumption = energy_demand_battery - battery_energy['energy'].iloc[i]
            battery_energy['energy'].iloc[i+1] = 0

    else:
        if battery_energy['energy'].iloc[i] < battery_capacity:
            battery_energy['energy'].iloc[i+1] = battery_energy['energy'].iloc[i] + (cfpv_generated-consumption)
            if battery_energy['energy'].iloc[i+1] > battery_capacity:
                battery_energy['energy'].iloc[i+1] = battery_capacity
    
plt.plot(battery_energy)

#Silicon 
battery_energy = pd.DataFrame(np.zeros(1440), columns=['energy'])
battery_energy['energy'].iloc[0] = battery_capacity*(SoC_battery/100)
for i in range(len(pv_profile)-1):
    p_generated = pv_profile['P'].iloc[i]/60
    consumption = load_profile['mult'].iloc[i]/60
    if consumption > p_generated:
        energy_demand_battery = (consumption - p_generated)

        if battery_energy['energy'].iloc[i] > energy_demand_battery:
            battery_energy['energy'].iloc[i+1] = battery_energy['energy'].iloc[i] - energy_demand_battery

        else:
            grid_consumption = energy_demand_battery - battery_energy['energy'].iloc[i]
            battery_energy['energy'].iloc[i+1] = 0

    else:
        if battery_energy['energy'].iloc[i] < battery_capacity:
            battery_energy['energy'].iloc[i+1] = battery_energy['energy'].iloc[i] + (p_generated-consumption)
            if battery_energy['energy'].iloc[i+1] > battery_capacity:
                battery_energy['energy'].iloc[i+1] = battery_capacity




plt.plot(battery_energy)
plt.plot(grid_consumption)
plt.plot(battery_energy)
plt.plot(consumption)
plt.plot(load_profile['mult'])
plt.plot(pv_profile[['P','P_CFPV']])
plt.savefig(f'{self_consumption_file}')
plt.show()
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants and setup
kWp = 5  # System capacity in kW peak
efficiency_base = 0.18  # Base efficiency at standard test conditions
temperature_coefficient_PV1 = -0.004  # Efficiency change per degree for PV1
temperature_coefficient_PV2 = -0.002  # Efficiency change per degree for PV2
STC_temperature = 25  # Standard test condition temperature

# Generating synthetic data
irradiance = np.linspace(0, 1000, 50)  # From 0 to 1 sun (1000 W/m^2)
temperature = np.linspace(15, 35, 50)  # Temperature range
I, T = np.meshgrid(irradiance, temperature)

# Functions to calculate efficiency based on temperature for two different materials
def calculate_efficiency(temp, temp_coeff):
    return efficiency_base + (temp - STC_temperature) * temp_coeff

# Calculate the power output for each material
def calculate_power_output(irradiance, efficiency):
    return kWp * (irradiance / 1000) * efficiency

# Calculate efficiencies for each material at each temperature
efficiency_PV1 = calculate_efficiency(T, temperature_coefficient_PV1)
efficiency_PV2 = calculate_efficiency(T, temperature_coefficient_PV2)

# Calculate power outputs
power_output_PV1 = calculate_power_output(I, efficiency_PV1)
power_output_PV2 = calculate_power_output(I, efficiency_PV2)

# Define the standard test condition (STC) PCE for both materials
STC_PCE = 0.18  # STC efficiency at 1000 W/m^2 and 25°C

# Calculate relative PCE for each material
relative_PCE_PV1 = efficiency_PV1 / STC_PCE
relative_PCE_PV2 = efficiency_PV2 / STC_PCE

# Adjust relative PCE for visualization purposes
low_light_performance_boost_PV1 = 1.1
relative_PCE_PV1[I < 200] *= low_light_performance_boost_PV1

# Plotting the enhanced 3D plot with relative PCE visualization
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot for material PV1 with relative PCE as color
surf1 = ax.plot_surface(I, T, power_output_PV1, facecolors=plt.cm.viridis(relative_PCE_PV1), 
                        rstride=1, cstride=1, alpha=0.8)

# Plot for material PV2 with relative PCE as color
surf2 = ax.plot_surface(I, T, power_output_PV2, facecolors=plt.cm.viridis(relative_PCE_PV2), 
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

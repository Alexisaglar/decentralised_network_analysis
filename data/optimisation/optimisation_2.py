import numpy as np
import matplotlib.pyplot as plt
from pypower.api import case33bw, runpf, ppoption
from pyswarms.discrete import BinaryPSO
from copy import deepcopy

# Load case
ppc = case33bw()

# Define power flow options
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=1)

# Get load bus indices (all buses are load buses in case33bw)
load_bus_indices = np.arange(len(ppc['bus']))

# Create Generation and Load Profiles (example, replace with actual data)
hours = 24
load = [0.1, 0.18, 0.2, 0.18, 0.15, 0.24, 0.5, 0.6, 0.55, 0.48, 0.45, 0.42, 0.4, 0.4, 0.5, 0.6, 0.8, 0.85, 0.9, 0.8, 0.7, 0.6, 0.3, 0.2] 
load_profile = np.array(load).reshape(-1, 1) /50
print(load_profile)

generation = [0, 0, 0, 0, 0, 0, 0.25, 1, 2, 3.25, 4.5, 5, 4.5, 3.25, 2, 1, 0.5, 0.25, 0, 0, 0, 0, 0, 0]  
generation_profile = np.array(generation).reshape(-1, 1) /20
print(generation_profile)

class BinaryPSO:
    def __init__(self, n_particles, dimensions, objective_func, max_iter=100):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.objective_func = objective_func
        self.max_iter = max_iter
        self.gbest_value = float('inf')
        self.gbest_position = np.zeros(self.dimensions)
        self.particles_position = np.random.randint(2, size=(self.n_particles, self.dimensions))
        self.pbest_position = self.particles_position
        self.pbest_value = np.array([float('inf')] * self.n_particles)
        self.velocity = np.zeros((self.n_particles, self.dimensions))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_velocity(self, w, c1, c2):
        for i in range(self.n_particles):
            for d in range(self.dimensions):
                r1, r2 = np.random.rand(2)
                cognitive_vel = c1 * r1 * (self.pbest_position[i, d] - self.particles_position[i, d])
                social_vel = c2 * r2 * (self.gbest_position[d] - self.particles_position[i, d])
                self.velocity[i, d] = w * self.velocity[i, d] + cognitive_vel + social_vel

    def update_position(self):
        for i in range(self.n_particles):
            for d in range(self.dimensions):
                if np.random.rand() < self.sigmoid(self.velocity[i, d]):
                    self.particles_position[i, d] = 1
                else:
                    self.particles_position[i, d] = 0

    def evaluate(self):
        for i in range(self.n_particles):
            fitness = self.objective_func(self.particles_position[i])
            if fitness < self.pbest_value[i]:
                self.pbest_value[i] = fitness
                self.pbest_position[i] = self.particles_position[i].copy()
            if fitness < self.gbest_value:
                self.gbest_value = fitness
                self.gbest_position = self.particles_position[i].copy()

    def run(self, w=0.7, c1=1.4, c2=1.4):
        for iteration in range(self.max_iter):
            self.update_velocity(w, c1, c2)
            self.update_position()
            self.evaluate()
        return self.gbest_position, self.gbest_value

# Check constraints function
def check_constraints(result):
    # Check if voltage limits are violated
    if any(bus[7] < bus[12] or bus[7] > bus[11] for bus in result['bus']):
        return False
    # Check if generator limits are violated
    for gen in result['gen']:
        if not (gen[9] <= gen[1] <= gen[8]) or not (gen[4] <= gen[2] <= gen[3]):
            return False
    return True

# Objective function considering generation and load profiles
def objective_function(x):
    total_power_losses = 0
    for hour in range(24):  # Assuming 24 hours for daily profiles
        ppc_temp = deepcopy(ppc)

        # Adjust load and subtract generation for the current hour at each bus
        for bus_idx in load_bus_indices:
            load_change = ppc_temp['bus'][bus_idx, 2] * (load_profile[hour]+1)
            generation_change = 0
            # If this bus is allowed to have generation, subtract the generation profile
            if x[bus_idx] == 1:
                generation_change = generation_profile[hour]  # Assuming same generation profile for all buses with generation
            net_load = load_change - generation_change
            ppc_temp['bus'][bus_idx, 2] = net_load

        # Run power flow
        result, success = runpf(ppc_temp, ppopt)
        
        # Calculate power losses
        if success and check_constraints(result):
            power_losses = sum(result['branch'][:, 13])  # Sum of real power losses in all branches
            total_power_losses += power_losses
        else:
            # Return a large number if constraints are violated or power flow fails
            return float('inf') 

    return total_power_losses


# PSO parameters
n_particles = 20
dimensions = len(load_bus_indices)
max_iter = 50

# Initialize and run BPSO
bpso = BinaryPSO(n_particles, dimensions, objective_function, max_iter)
best_position, best_value = bpso.run(max_iter)
print(best_position)

total_losses = []
total_losses_without_pv = []
total_load_consumption = []
voltage_profiles = [[] for _ in range(len(ppc['bus']))]
voltage_profiles_without_pv = [[] for _ in range(len(ppc['bus']))]

for hour in range(hours):
    ppc_temp = deepcopy(ppc)
    for bus_idx in load_bus_indices:
        load_change = ppc_temp['bus'][bus_idx, 2] * (load_profile[hour]+1)
        load_change_without_pv = ppc_temp['bus'][bus_idx, 2] * (load_profile[hour]+1)
        ppc['bus'][bus_idx, 2] = load_change_without_pv
        generation_change = 0
        if best_position[bus_idx] == 1:
            generation_change = generation_profile[hour]
        net_load = load_change - generation_change
        ppc_temp['bus'][bus_idx, 2] = net_load
    # Run power flow
    result, _ = runpf(ppc_temp, ppopt)
    result_without_pv, _ = runpf(ppc, ppopt)

    # Store voltage values
    for bus_idx in load_bus_indices:
        voltage_profiles[bus_idx].append(result['bus'][bus_idx, 7])
        voltage_profiles_without_pv[bus_idx].append(result_without_pv['bus'][bus_idx, 7])

    # Sum the real power losses in all branches
    losses = sum(result['branch'][:, 13] + result['branch'][:, 15])
    losses_without_pv = sum(result_without_pv['branch'][:, 13] + result_without_pv['branch'][:, 15])
    total_losses.append(losses)
    total_losses_without_pv.append(losses_without_pv)
    total_load_consumption.append(load_change_without_pv)


# Graph Total Power Losses over 24 hours
plt.figure(figsize=(10, 5))
plt.plot(total_losses, 'o-')
plt.plot(total_losses_without_pv, 'o-')
plt.title('Total Power Losses over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Total Power Losses (MW)')
plt.grid(True)
plt.show()

# Now we create the box plots
plt.figure(figsize=(15, 8))
plt.boxplot(voltage_profiles)
plt.title('Voltage Profile Box Plot for Each Bus Over 24 Hours')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.xticks(np.arange(1, len(ppc['bus']) + 1), np.arange(1, len(ppc['bus']) + 1))  # Ensure that x-ticks match the bus numbers
plt.grid(True)
plt.show()

hours = np.arange(1, 25)  # 24-hour period
total_pv_generation = [sum(g) for g in generation_profile]  # Summing up generation for all PV systems per hour
# total_load_consumption = [sum(l) for l in load_profile]  # Summing up load for all buses per hour

#plot both generation and consumption in during the day
plt.figure(figsize=(12, 6))
# Plotting generation
plt.plot(hours, total_pv_generation, label='PV Generation (MW)', color='green', marker='o')
# Plotting load consumption
plt.plot(hours, total_load_consumption, label='Load Consumption (MW)', color='red', marker='x')
plt.title('PV Generation and Load Consumption over 24 Hours')
plt.xlabel('Hour of the Day')
plt.ylabel('Power (MW)')
plt.xticks(hours)  # Ensure that x-ticks match the hours
plt.grid(True)
plt.legend()
plt.show()

# Plotting
fig, ax = plt.subplots(figsize=(15, 5))

# Create a bar graph
bus_numbers = np.arange(1, len(best_position) + 1)  # Bus numbers from 1 to 33
ax.bar(bus_numbers, best_position, color='green')

# Set the title and labels
ax.set_title('PV Installation on Buses (from BPSO Optimization)')
ax.set_xlabel('Bus Number')
ax.set_ylabel('PV Installed (0 = No, 1 = Yes)')

# Show the plot
plt.xticks(bus_numbers)  # Ensure all bus numbers are shown
plt.tight_layout()
plt.show()

# Now we create the box plots
plt.figure(figsize=(15, 8))

# Plot with PV
plt.boxplot(voltage_profiles, positions=np.arange(1, 2*len(voltage_profiles)+1, 2), widths=0.35, patch_artist=True, boxprops=dict(facecolor='blue', color='blue'), medianprops=dict(color='white'), labels=['With PV']*len(voltage_profiles))

# Plot without PV
plt.boxplot(voltage_profiles_without_pv, positions=np.arange(2, 2*len(voltage_profiles_without_pv)+2, 2), widths=0.35, patch_artist=True, boxprops=dict(facecolor='red', color='red'), medianprops=dict(color='white'), labels=['Without PV']*len(voltage_profiles_without_pv))

plt.title('Voltage Profile Box Plot for Each Bus Over 24 Hours')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')

# Custom x-axis labels to include both 'With PV' and 'Without PV'
ax = plt.gca()
ax.set_xticks(np.arange(1.5, 2*len(voltage_profiles)+1, 2))
ax.set_xticklabels(np.arange(1, len(voltage_profiles)+1))
ax.set_xlim(0, 2*len(voltage_profiles)+1)

plt.grid(True)
plt.legend(['With PV', 'Without PV'])
plt.show()
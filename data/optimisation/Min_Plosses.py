import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypower.api import case33bw, runpf, ppoption
from pyswarms.discrete import BinaryPSO
from copy import deepcopy
import seaborn as sns
from joblib import Parallel, delayed



# Load case
ppc = case33bw()

# Define power flow options
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)

# Get load bus indices (all buses are load buses in case33bw)
load_bus_indices = np.arange(len(ppc['bus']))


# Create Generation and Load Profiles (example, replace with actual data)
hours = 24  
load = [0.1, 0.18, 0.2, 0.18, 0.15, 0.24, 0.5, 0.6, 0.55, 0.48, 0.45, 0.42, 0.4, 0.4, 0.5, 0.6, 0.8, 0.85, 1, 0.85, 0.7, 0.6, 0.3, 0.2] 
load_profile = np.array(load).reshape(-1, 1) 
print(load_profile)

generation = pd.read_csv('~/decentralised_network_analysis/data/optimisation/hourly_Jan.csv')
# generation = [0, 0, 0, 0, 0, 0, 0.25, 1, 2, 3.25, 4.5, 5, 4.5, 3.25, 2, 1, 0.5, 0.25, 0, 0, 0, 0, 0, 0]  
generation_profile_silicon = np.array(generation['P']).reshape(-1, 1) /50000
generation_profile_LLE = np.array(generation['P_CFPV']).reshape(-1, 1) /50000
print(generation_profile_silicon)
print(generation_profile_LLE)

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
        self.gbest_value_history = []

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
            # fitnesses = Parallel(n_jobs=-1)(delayed(self.objective_func)(position) for position in self.particles_position)
            # for i, fitness in enumerate(fitnesses):
            if fitness < self.pbest_value[i]:
                self.pbest_value[i] = fitness
                self.pbest_position[i] = self.particles_position[i].copy()
            if fitness < self.gbest_value:
                self.gbest_value = fitness
            self.gbest_position = self.particles_position[i].copy()
        self.gbest_value_history.append(self.gbest_value)  # Update history


    def run(self, c1=1.4, c2=1.4):
        w_max = 0.9 # Initial inertia weight
        w_min = 0.6  # Final inertia weight 

        for iteration in range(self.max_iter):
            w = w_max - (w_max - w_min) * iteration / self.max_iter  # Linearly decreasing w
            self.update_velocity(w, c1, c2)
            self.update_position()
            self.evaluate()
            
            if iteration > 50 and abs(self.gbest_value_history[-1] - self.gbest_value_history[-50]) < 1e-6:
                print(f"Stopping early at iteration {iteration} due to small changes in global best value.")
                break
        print(self.gbest_position)
        return self.gbest_position, self.gbest_value

    def plot_convergence_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.gbest_value_history, linewidth=2)
        plt.title("BPSO Convergence Curve")
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()

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
            load_change = ppc_temp['bus'][bus_idx, 2] * (load_profile[hour])
            reactive_change = ppc_temp['bus'][bus_idx, 3] * (load_profile[hour])
            # load_change = load_profile[hour]
            generation_change = 0
            # If this bus is allowed to have generation, subtract the generation profile
            if x[bus_idx] == 1:
                generation_change = generation_profile_silicon[hour]  # Assuming same generation profile for all buses with generation
            net_load = load_change - generation_change
            ppc_temp['bus'][bus_idx, 2] = net_load
            ppc_temp['bus'][bus_idx, 3] = reactive_change
            
        # Run power flow
        result, success = runpf(ppc_temp, ppopt)
        
        # Calculate power losses
        if success and check_constraints(result):
            power_losses = sum(result['branch'][:, 13] + result['branch'][:, 15])  # Sum of real power losses in all branches
            total_power_losses += power_losses
            
        else:
            # Return a large number if constraints are violated or power flow fails
            return float('inf') 
    
    return total_power_losses


# PSO parameters
n_particles = 30
dimensions = len(load_bus_indices)
max_iter = 500
n_runs = 1
bus_frequency = np.zeros(dimensions)  # Added: Track the frequency of each bus being selected for PV installation

# Record the best positions and values for each run
all_best_positions = []
all_best_values = []
all_gbest_values = []

for _ in range(n_runs):
    bpso = BinaryPSO(n_particles, dimensions, objective_function, max_iter)
    best_position, best_value = bpso.run(max_iter)
    all_best_positions.append(best_position)
    all_best_values.append(best_value)
    all_gbest_values.append(bpso.gbest_value_history)  # Update this line to append the gbest_value_history

    # Increment the frequency for the buses where PV is installed in the best position of this run
    bus_frequency += best_position

def evaluate_losses_for_profile(x, generation_profile):
    total_power_losses = 0
    for hour in range(24):  # Assuming 24 hours for daily profiles
        ppc_temp = deepcopy(ppc)
        # Adjust load and subtract generation for the current hour at each bus
        for bus_idx in load_bus_indices:
            load_change = ppc_temp['bus'][bus_idx, 2] * (load_profile[hour])
            reactive_change = ppc_temp['bus'][bus_idx, 3] * (load_profile[hour])
            # load_change = load_profile[hour]
            generation_change = 0
            # If this bus is allowed to have generation, subtract the generation profile
            if x[bus_idx] == 1:
                generation_change = generation_profile[hour]  # Assuming same generation profile for all buses with generation
            net_load = load_change - generation_change
            ppc_temp['bus'][bus_idx, 2] = net_load
            ppc_temp['bus'][bus_idx, 3] = reactive_change
            
        # Run power flow
        result, success = runpf(ppc_temp, ppopt)
        
        # Calculate power losses
        if success and check_constraints(result):
            power_losses = sum(result['branch'][:, 13] + result['branch'][:, 15])  # Sum of real power losses in all branches
            total_power_losses += power_losses
            
        else:
            # Return a large number if constraints are violated or power flow fails
            return float('inf') 
    
    return total_power_losses

total_losses_silicon = evaluate_losses_for_profile(best_position, generation_profile_silicon)
total_losses_LLE = evaluate_losses_for_profile(best_position, generation_profile_LLE)

# Now you can compare total_losses_silicon and total_losses_LLE
print("Total losses with Silicon profile:", total_losses_silicon)
print("Total losses with LLE profile:", total_losses_LLE)


# Find the overall best solution
best_of_best_index = np.argmin(all_best_values)
overall_best_position = all_best_positions[best_of_best_index]
print("Overall Best Position:", overall_best_position)
print("With a loss value of:", all_best_values[best_of_best_index])

total_losses_Si_PV = []
total_losses_LLE = []
total_losses_without_pv = []

total_load_consumption_Si_PV = []
total_load_consumption_LLE = []
total_load_consumption_without_pv = []

total_load_Si_PV = []
total_load_LLE = []
total_load_without_pv = []

voltage_profiles_Si_PV = [[] for _ in range(len(ppc['bus']))]
voltage_profiles_LLE = [[] for _ in range(len(ppc['bus']))]
voltage_profiles_without_pv = [[] for _ in range(len(ppc['bus']))]

for hour in range(hours):
    ppc_Si_PV = deepcopy(ppc)
    ppc_without_pv = deepcopy(ppc)
    ppc_LLE_PV = deepcopy(ppc)
    for bus_idx in load_bus_indices:
        generation_change_Si_PV = 0
        generation_change_LLE_PV = 0
        load_change = ppc_without_pv['bus'][bus_idx, 2] * (load_profile[hour])
        load_change_without_pv = ppc_without_pv['bus'][bus_idx, 2] * (load_profile[hour])
        reactive_change = ppc_without_pv['bus'][bus_idx, 3] * (load_profile[hour])
        if overall_best_position[bus_idx] == 1:
            generation_change_Si_PV= generation_profile_silicon[hour]
            generation_change_LLE_PV = generation_profile_LLE[hour]

        # Make changes of values in data without PV
        ppc_without_pv['bus'][bus_idx, 2] = load_change_without_pv
        ppc_without_pv['bus'][bus_idx, 3] = reactive_change
        
        #Calculate net load for different PV profiles
        net_load_LLE = load_change - generation_change_LLE_PV
        net_load_Si_PV = load_change - generation_change_Si_PV

        # Make changes in active and reactive power for Si-PV values
        ppc_Si_PV['bus'][bus_idx, 2] = net_load_Si_PV
        ppc_Si_PV['bus'][bus_idx, 3] = reactive_change

        # Make changes in active and reactive power for Si-PV values
        ppc_LLE_PV['bus'][bus_idx, 2] = net_load_LLE 
        ppc_LLE_PV['bus'][bus_idx, 3] = reactive_change
    # Run power flow

    result_LLE_PV, _ = runpf(ppc_LLE_PV, ppopt)
    result_Si_PV, _ = runpf(ppc_Si_PV, ppopt)
    result_without_pv, _ = runpf(ppc_without_pv, ppopt)

    # Store voltage values
    for bus_idx in load_bus_indices:
        voltage_profiles_Si_PV[bus_idx].append(result_Si_PV['bus'][bus_idx, 7])
        voltage_profiles_LLE[bus_idx].append(result_LLE_PV['bus'][bus_idx, 7])
        voltage_profiles_without_pv[bus_idx].append(result_without_pv['bus'][bus_idx, 7])

    # Sum the real power losses in all branches

    total_load_Si_PV = (sum(ppc_Si_PV['bus'][:,2]))
    total_load_LLE = (sum(ppc_LLE_PV['bus'][:,2]))
    total_load_without_pv = (sum(ppc_without_pv['bus'][:,2]))

    total_load_consumption_Si_PV.append(total_load_Si_PV)
    total_load_consumption_LLE.append(total_load_LLE)
    total_load_consumption_without_pv.append(total_load_without_pv)

    # Total losses calculation
    losses_Si_PV = sum(result_Si_PV['branch'][:, 13] + result_Si_PV['branch'][:, 15])
    losses_LLE = sum(result_LLE_PV['branch'][:, 13] + result_LLE_PV['branch'][:, 15])
    losses_without_pv = sum(result_without_pv['branch'][:, 13] + result_without_pv['branch'][:, 15])

    total_losses_Si_PV.append(losses_Si_PV)
    total_losses_LLE.append(losses_LLE)
    total_losses_without_pv.append(losses_without_pv) 

print('Total losses without PV:\n', sum(total_losses_without_pv))
print('Total losses with Si-PV:\n', sum(total_losses_Si_PV))
print('Total losses with LLE-PV:\n', sum(total_losses_LLE))

# Creating a figure with two subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))

# Heatmap for Bus Voltage Profiles Over Time
sns.heatmap(voltage_profiles_Si_PV, cmap="viridis", ax=ax1)
ax1.set_xlabel('Hour of the Day')
ax1.set_ylabel('Bus number')
ax1.set_title('Voltage Profile with Si-PV')
ax1.legend()

# Heatmap for Bus Voltage Profiles Over Time
sns.heatmap(voltage_profiles_LLE, cmap="viridis", ax=ax2)
ax2.set_xlabel('Hour of the Day')
ax2.set_ylabel('Bus number')
ax2.set_title('Voltage Profile with LLE-PV')
ax2.legend()

# Heatmap for Bus Voltage Profiles Over Time
sns.heatmap(voltage_profiles_without_pv, cmap="viridis", ax=ax3)
ax3.set_xlabel('Hour of the Day')
ax3.set_ylabel('Bus number')
ax3.set_title('Voltage Profile without PV')
ax3.legend()
plt.show()

# Bar Chart for Frequency of Bus Selection for PV Generation
plt.figure(figsize=(15, 5))
plt.bar(range(1, 34), best_position)
plt.title('Buses selected for this solution')
plt.xlabel('Bus Number')
plt.ylabel('Selection')
plt.grid(True)
plt.xticks(range(1, 34))  # Assuming bus numbers are 1-indexed
plt.show()

# Creating a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Plotting the loads on the left subplot (ax1)
ax1.plot(np.arange(hours), total_load_consumption_Si_PV, label='Total Load with Si-PV', color='blue')
ax1.plot(np.arange(hours), total_load_consumption_LLE, label='Total Load with LLE-PV', color='green')
ax1.plot(np.arange(hours), total_load_consumption_without_pv, label='Load without PV', color='red')
ax1.set_ylabel('Load (kW)')
ax1.set_xlabel('Time (Hours)')
ax1.set_title('Load')
ax1.legend()
ax1.grid(True)

# Plotting the losses on the right subplot (ax2)
ax2.plot(np.arange(hours), total_losses_without_pv, label='Total losses', color='red', linestyle='--')
ax2.plot(np.arange(hours), total_losses_Si_PV, label='Losses with Si-PV', color='blue', linestyle='--')
ax2.plot(np.arange(hours), total_losses_LLE, label='Losses with LLE-PV', color='green', linestyle='--')
ax2.set_ylabel('Power Loss (MW)')
ax2.set_xlabel('Time (hours)')
ax2.set_title('Losses')
ax2.legend()
ax2.grid(True)

# Display the plots
plt.show()


plt.figure(figsize=(10, 6))
for run_number, gbest_values in enumerate(all_gbest_values):
    plt.plot(gbest_values, label=f'Run {run_number + 1}')
plt.title("BPSO Convergence Curves for All Runs")
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Value')
plt.legend()
plt.grid(True)
plt.show()
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
load_profile = np.array(load).reshape(-1, 1) / len(ppc['bus'])

generation = [0, 0, 0, 0, 0, 0, 0.25, 1, 2, 3.25, 4.5, 5, 4.5, 3.25, 2, 1, 0.5, 0.25, 0, 0, 0, 0, 0, 0]  
generation_profile = np.array(generation).reshape(-1, 1) / len(ppc['bus'])

  # Define the number of particles and dimensions
n_particles = 20
dimensions = len(load_bus_indices)

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
    for hour in range(hours):
        ppc_temp = deepcopy(ppc)
        # Adjust load and generation for the current hour
        for bus_idx, load_factor in enumerate(load_profile[hour]):
            ppc_temp['bus'][bus_idx, 2] *= load_factor
        for gen_idx, gen_factor in enumerate(generation_profile[hour]):
            ppc_temp['gen'][gen_idx, 1] *= gen_factor
        
        # Add PV penetration at the chosen bus (if applicable)
        best_bus_idx = np.argmax(x)
        if x[best_bus_idx] == 1:
            ppc_temp['bus'][load_bus_indices[best_bus_idx], 2] -= 0.1
        
        # Run power flow
        result, success = runpf(ppc_temp, ppopt)
        # Calculate power losses
        if success and check_constraints(result):
            power_losses = sum(result['branch'][:, 13])  # Sum of real power losses in all branches
            total_power_losses += power_losses
        else:
            return float('inf')  # Return a large number if constraints are violated or power flow fails
    
    return total_power_losses

# PSO parameters
n_particles = 20
dimensions = len(load_bus_indices)
max_iter = 50

# Initialize and run BPSO
bpso = BinaryPSO(n_particles, dimensions, objective_function, max_iter)
best_position, best_value = bpso.run(max_iter)

# Apply the best PV penetration to the system for each hour and record total power losses
total_power_losses_over_time = []
voltages_over_time = []
for hour in range(hours):
    ppc_temp = deepcopy(ppc)
    # Adjust load and generation for the current hour
    for bus_idx, load_factor in enumerate(load_profile[hour]):
        ppc_temp['bus'][bus_idx, 2] *= load_factor
    for gen_idx, gen_factor in enumerate(generation_profile[hour]):
        ppc_temp['gen'][gen_idx, 1] *= gen_factor
    
    # Add PV penetration at the chosen bus
    best_bus_idx = np.argmax(best_position)
    if best_position[best_bus_idx] == 1:
        ppc_temp['bus'][load_bus_indices[best_bus_idx], 2] -= 0.1
    
    # Run power flow
    result, success = runpf(ppc_temp, ppopt)
    if success:
        power_losses = sum(result['branch'][:, 13])  # Sum of real power losses in all branches
        total_power_losses_over_time.append(power_losses)
        voltages_over_time.append([bus[7] for bus in result['bus']])


# Plotting
plt.figure(figsize=(21, 7))

plt.subplot(1, 3, 1)
plt.bar(load_bus_indices, best_position, color='orange')
plt.xlabel('Bus Number')
plt.ylabel('PV Penetration (Binary)')
plt.title('Optimized PV Penetration at Each Load Bus')

plt.subplot(1, 3, 2)
plt.plot(range(hours), total_power_losses_over_time, marker='o', color='green')
plt.xlabel('Hour')
plt.ylabel('Total Power Losses (MW)')
plt.title('Total Power Losses Over Time')

plt.subplot(1, 3, 3)
# Choose an hour to display voltage profile, or take the average
average_voltages = np.mean(voltages_over_time, axis=0)
plt.plot(load_bus_indices, average_voltages, marker='o', linestyle='-', color='blue')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage Profile at Each Bus')

plt.tight_layout()
plt.show()





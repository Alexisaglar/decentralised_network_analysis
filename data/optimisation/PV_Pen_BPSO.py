import numpy as np
from pypower.api import case33bw, runpf, ppoption
from copy import deepcopy
import matplotlib.pyplot as plt

# Load case
ppc = case33bw()

# Define power flow options
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=1)

# Get load bus indices (all buses are load buses in case33bw)
load_bus_indices = np.arange(len(ppc['bus']))

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

def check_constraints(result):
    # Check if voltage limits are violated
    if any(bus[7] < bus[12] or bus[7] > bus[11] for bus in result['bus']):
        return False
    # Check if generator limits are violated
    for gen in result['gen']:
        if not (gen[9] <= gen[1] <= gen[8]) or not (gen[4] <= gen[2] <= gen[3]):
            return False
    return True

def objective_function(x):
    ppc_temp = deepcopy(ppc)  # Use deepcopy instead of np.copy
    # Add PV penetration at the chosen bus
    bus_idx = np.argmax(x)  # Choose the bus with a PV system (binary decision)
    ppc_temp['bus'][load_bus_indices[bus_idx], 2] -= 0.5  # Subtract 5 MW from the load
    # Run power flow
    result, success = runpf(ppc_temp, ppopt)
    # Calculate power losses
    if success and check_constraints(result):
        power_losses = sum(result['branch'][:, 13])  # Sum of real power losses in all branches
        return power_losses
    else:
        return float('inf')  # Return a large number if constraints are violated or power flow fails

# PSO parameters
n_particles = 20
dimensions = len(load_bus_indices)
max_iter = 50

# Initialize and run BPSO
bpso = BinaryPSO(n_particles, dimensions, objective_function, max_iter)
best_position, best_value = bpso.run()

print("Best Position for PV Penetration:", best_position)
print("Best Power Loss Value:", best_value)

# Apply the best PV penetration to the system
ppc_best = deepcopy(ppc)
best_bus_idx = np.argmax(best_position)
ppc_best['bus'][load_bus_indices[best_bus_idx], 2] -= 0.5

# Run power flow for the best solution
result, success = runpf(ppc_best, ppopt)
# Plotting
voltages = [bus[7] for bus in result['bus']]

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.bar(load_bus_indices, best_position, color='orange')
plt.xlabel('Bus Number')
plt.ylabel('PV Penetration (Binary)')
plt.title('Optimized PV Penetration at Each Load Bus')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(ppc_best['bus'])+1), voltages, marker='o', color='green')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage at Each Bus with Optimized PV Penetration')

plt.tight_layout()
plt.show()
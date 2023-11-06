import numpy as np
import random
from pypower.api import case33bw, runpf, ppoption
import matplotlib.pyplot as plt
import pyswarms as ps
from copy import deepcopy

# Load case
ppc = case33bw()

# Define power flow options
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=1)  # Change verbosity to 0 to reduce output

# Get load bus indices (all buses are load buses in case33bw)
load_bus_indices = np.arange(len(ppc['bus']))

# Define the number of Monte Carlo runs
n_runs = 10

# Record the best cost and position for each run
best_losses = np.zeros(n_runs)
best_positions = np.zeros((n_runs, len(load_bus_indices)))

def check_constraints(result):
    # Check if voltage limits are violated
    if any(bus[7] < bus[12] or bus[7] > bus[11] for bus in result['bus']):
        return False
    # Check if generator limits are violated
    for gen in result['gen']:
        if not (gen[8] <= gen[1] <= gen[9]) or not (gen[4] <= gen[2] <= gen[5]):
            return False
    return True

def objective_function(x):
    power_losses = np.zeros(x.shape[0])
    for idx, particle in enumerate(x):
        ppc_temp = deepcopy(ppc)  # Use deepcopy instead of np.copy
        # Add PV penetration
        for i, val in enumerate(particle):
            if val > 0.5:  # If particle dimension value is greater than 0.5, install 5 MW PV
                ppc_temp['bus'][load_bus_indices[i], 2] -= .005
        # Run power flow
        result, success = runpf(ppc_temp, ppopt)
        # Calculate power losses if power flow is successful and no constraints are violated
        if success and check_constraints(result):
            total_loss = sum(result['branch'][:, 13])  # The total real power loss in the system
            power_losses[idx] = total_loss
        else:
            power_losses[idx] = 1000  # Large value if constraints are violated (acts as a penalty)
    return power_losses

# PSO parameters
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.zeros(len(load_bus_indices)), np.ones(len(load_bus_indices)))

for run in range(n_runs):
    # Initialize the swarm
    optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=len(load_bus_indices), options=options, bounds=bounds)

    # Perform optimization
    loss, pos = optimizer.optimize(objective_function, iters=50)

    # Store the best loss and position
    best_losses[run] = loss
    best_positions[run, :] = pos

# Analyze the results
average_loss = np.mean(best_losses)
std_dev_loss = np.std(best_losses)

print(f"Average optimized power loss: {average_loss}")
print(f"Standard deviation of optimized power loss: {std_dev_loss}")

# Find the best solution from all Monte Carlo runs
best_run_idx = np.argmin(best_losses)
best_solution = best_positions[best_run_idx, :]

# Apply the best PV penetration to the system
ppc_best = deepcopy(ppc)
for i, val in enumerate(best_solution):
    if val > 0.5:
        ppc_best['bus'][load_bus_indices[i], 2] -= 0.005

# Run power flow for the best solution
result, success = runpf(ppc_best, ppopt)

# Plotting
voltages = [bus[7] for bus in result['bus']]

plt.figure(figsize=(14, 7))

# Count the frequency of PV penetration at each bus
pv_penetration_frequency = np.sum(best_positions > 0.5, axis=0)

plt.subplot(1, 3, 1)
plt.bar(load_bus_indices, pv_penetration_frequency, color='lightblue')
plt.xlabel('Bus Number')
plt.ylabel('Frequency of PV Penetration')
plt.title('Frequency of PV Penetration at Each Load Bus')

plt.subplot(1, 3, 2)
plt.bar(load_bus_indices, best_solution > 0.5, color='orange')
plt.xlabel('Bus Number')
plt.ylabel('PV Penetration (5 MW Units)')
plt.title('Best Optimized PV Penetration at Each Load Bus')

plt.subplot(1, 3, 3)
plt.plot(range(1, len(ppc_best['bus'])+1), voltages, marker='o', color='green')
plt.xlabel('Bus Number')
plt.ylabel('Voltage (p.u.)')
plt.title('Voltage at Each Bus with Optimized Power Losses')

plt.tight_layout()
plt.show()

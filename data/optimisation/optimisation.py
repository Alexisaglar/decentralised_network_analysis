import numpy as np
from pypower.api import runpf, ppoption, loadcase
from pyswarm import pso
import matplotlib.pyplot as plt
import random
from copy import deepcopy

PD = 2  # column index for active power demand in bus matrix

def fitness_function(x, mpc, ppopt):
    gen_count = mpc['gen'].shape[0]
    
    # Set variables
    mpc['gen'][:, 1] = x[:gen_count]  # PG
    mpc['gen'][:, 5] = x[gen_count:2*gen_count]  # VG
    mpc['gen'][:, 8] = x[2*gen_count:3*gen_count]  # Tc
    mpc['gen'][:, 2] = x[3*gen_count:4*gen_count]  # Qc

    result, success = runpf(mpc, ppopt)

    if not success:
        return 1e10

  # Check for voltage constraints
    voltage_magnitudes = result['bus'][:, 7]
    if any(voltage < 0.95 or voltage > 1.05 for voltage in voltage_magnitudes):
        return 1e10
    return np.sum(result['branch'][:, 13])

def main():
    mpc = loadcase("/opt/homebrew/lib/python3.11/site-packages/pypower/case30.py")
    original_mpc = deepcopy(mpc)  # Backup the original state

    gen_count = mpc['gen'].shape[0]
    bus_count = mpc['bus'].shape[0]
    ppopt = ppoption(VERBOSE=0)  # Added the missing ppopt definition here.
    
    # Prepare bounds
    lb = np.concatenate([mpc['gen'][:, 9], 0.95 * mpc['gen'][:, 5], 0.95 * np.ones(gen_count), mpc['gen'][:, 4]])
    ub = np.concatenate([mpc['gen'][:, 8], 1.05 * mpc['gen'][:, 5], 1.05 * np.ones(gen_count), mpc['gen'][:, 3]])

    MAX_CONSECUTIVE_VIOLATIONS = 1
    consecutive_violations = 0
    pv_penetration_values = []
    pv_buses = set()

    while consecutive_violations < MAX_CONSECUTIVE_VIOLATIONS:
        if len(pv_buses) == bus_count:  # Break when all buses have a PV system
            break

        # Randomly select a bus to add PV
        bus_to_add_pv = random.choice([b for b in range(bus_count) if b not in pv_buses])
        pv_buses.add(bus_to_add_pv)

        # Add 1 MW of PV by reducing the load at the selected bus
        mpc['bus'][bus_to_add_pv, PD] -= 10

        # Optimize using PSO
        xopt, fopt = pso(fitness_function, lb, ub, args=(mpc, ppopt), swarmsize=50, maxiter=500, minstep=1e-4)
        
        # Check if the optimization result is feasible
        if fopt > 1e9:  # This indicates a violation (based on the return value in fitness_function)
            consecutive_violations += 1
        else:
            consecutive_violations = 0  # Reset if there's a feasible solution

        pv_penetration_values.append(len(pv_buses))
        
        # Run the optimized configuration to get the bus voltages
        mpc['gen'][:, 1] = xopt[:gen_count]  # PG
        mpc['gen'][:, 5] = xopt[gen_count:2*gen_count]  # VG
        mpc['gen'][:, 8] = xopt[2*gen_count:3*gen_count]  # Tc
        mpc['gen'][:, 2] = xopt[3*gen_count:4*gen_count]  # Qc
        result, _ = runpf(mpc, ppopt)
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(bus_count), result['bus'][:, 7])
        plt.xlabel("Bus Number")
        plt.ylabel("Voltage (p.u.)")
        plt.title(f"Voltage Magnitude per Bus with {len(pv_buses)} PV Systems")
        plt.grid(True)
        plt.show()

    # Plot the increase in PV penetration
    plt.figure(figsize=(10, 5))
    plt.plot(pv_penetration_values)
    plt.xlabel("Iterations")
    plt.ylabel("Number of PV Systems")
    plt.title("Increase in PV Penetration")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

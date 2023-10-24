import numpy as np
from pypower.api import runpf, ppoption, loadcase
from pyswarm import pso

def fitness_function(x):
    mpc = loadcase("case30")

    # Debug: Print mpc to check its contents
    print(mpc)

    gen_count = np.array(mpc['gen']).shape[0]
    bus_count = np.array(mpc['bus']).shape[0]

    # Set variables
    mpc['gen'][:, 1] = x[:gen_count]
    mpc['gen'][:, 5] = x[gen_count:2*gen_count]
    mpc['gen'][:, 8] = x[2*gen_count:3*gen_count]
    mpc['gen'][:, 2] = x[3*gen_count:4*gen_count]

    ppopt = ppoption(VERBOSE=0)
    result, success = runpf(mpc, ppopt)

    if not success:
        return 1e10

    return np.sum(result['branch'][:, 13])

def main():
    mpc = loadcase("case30")

    # Debug: Print mpc to check its contents
    print(mpc)

    gen_count = np.array(mpc['gen']).shape[0]
    bus_count = np.array(mpc['bus']).shape[0]

    # Lower and Upper bounds for PG, VG, Tc, and Qc
    lb = np.concatenate([mpc['gen'][:, 9], 0.9 * mpc['gen'][:, 5], 0.9 * np.ones(gen_count), mpc['gen'][:, 3]])
    ub = np.concatenate([mpc['gen'][:, 8], 1.1 * mpc['gen'][:, 5], 1.1 * np.ones(gen_count), mpc['gen'][:, 4]])

    # PSO optimization
    optimal_x, _ = pso(fitness_function, lb, ub, swarmsize=50, maxiter=500, minstep=1e-4)

    # Plot the voltage profile
    ppopt = ppoption(VERBOSE=0)
    result, success = runpf(mpc, ppopt)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(range(bus_count), result['bus'][:, 7])
    plt.xlabel("Bus Number")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage Magnitude per Bus after Optimal PV Penetration")
    plt.grid(True)
    plt.show()

    # ... (additional visualizations as needed)

if __name__ == "__main__":
    main()

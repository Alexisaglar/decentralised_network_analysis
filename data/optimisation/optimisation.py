import numpy as np
from pypower.api import case30, runpf
from pyswarm import pso

# Load case
ppc = case30()

def objective_function(x):
    Pg = x[:6]  # Active power outputs
    Vg = x[6:12]  # Voltage magnitudes
    Tc = x[12:18]  # Hypothetical transmission constraints
    Qc = x[18:24]  # Reactive power constraints

    # Update power system case with decision variables
    ppc["gen"][:, 1] = Pg  # Pgen
    ppc["gen"][:, 5] = Vg  # Vm

    # Power flow
    results, _ = runpf(ppc)
    cost = sum(results["gencost"][:, 4] * (Pg ** 2) + results["gencost"][:, 5] * Pg + results["gencost"][:, 6])

    # Penalty term for inequality constraints
    penalty = 0

    # For Pg
    for val in Pg:
        if val < ppc["gen"][:, 9][0] or val > ppc["gen"][:, 8][0]:  # Pmin and Pmax
            penalty += 1e6

    # For Vg
    for val in Vg:
        if val < 0.95 or val > 1.05:  # Voltage limits
            penalty += 1e6

    # For Tc (example limits: 0 and 100)
    for val in Tc:
        if val < 0 or val > 100:
            penalty += 1e6

    # For Qc (example limits: Qmin and Qmax)
    for val in Qc:
        if val < ppc["gen"][:, 3][0] or val > ppc["gen"][:, 4][0]:  # Qmin and Qmax
            penalty += 1e6

    # Penalty for equality constraints (power balance)
    if abs(sum(Pg) - sum(ppc["bus"][:, 2])) > 1e-3:  # 1e-3 is a small tolerance
        penalty += 1e6

    return cost + penalty

# Lower and Upper bounds
lb = np.concatenate([ppc["gen"][:, 9], [0.95]*6, [0]*6, ppc["gen"][:, 4]])
ub = np.concatenate([ppc["gen"][:, 8], [1.05]*6, [100]*6, ppc["gen"][:, 3]])

xopt, fopt = pso(objective_function, lb, ub, swarmsize=50, maxiter=30, minfunc=1e-8, minstep=1e-8)

print("Optimal solution:", xopt)
print("Objective value:", fopt)

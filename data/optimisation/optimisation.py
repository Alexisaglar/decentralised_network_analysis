import numpy as np
from pyswarm import pso
import pypower.api as pp
from pypower.case9 import case9

# Define the fitness function
def fitness_function(x, ppc):
    V = x[:len(Npq)]
    T = x[len(Npq):len(Npq)+len(Nt)]
    QG = x[len(Npq)+len(Nt):]

    # Calculate penalties based on provided equations
    penalty_V = np.sum([lambda_Vi * (Vi - V_lim_i)**2 for Vi, V_lim_i in zip(V, V_lim)])
    penalty_T = np.sum([lambda_Ti * (Ti - T_lim_i)**2 for Ti, T_lim_i in zip(T, T_lim)])
    penalty_QG = np.sum([lambda_Gi * (QGi - QG_lim_i)**2 for QGi, QG_lim_i in zip(QG, QG_lim)])
    
    return penalty_V + penalty_T + penalty_QG

# Load PyPower case
ppc = case9()

# Extract necessary data
Npq = list(range(len(ppc['bus'])))
Nt = list(range(len(ppc['branch'])))
Ng = list(range(len(ppc['gen'])))

# Define the constraints
# This is a basic assumption for the constraints, and it may need adjustments based on your specific problem.
lb = [0.9]*len(Npq) + [0.95]*len(Nt) + [0]*len(Ng)  # Lower bounds
ub = [1.1]*len(Npq) + [1.05]*len(Nt) + [100]*len(Ng)  # Upper bounds

# Sample penalty multipliers - These should be tuned or defined based on the problem specifics
lambda_Vi = 1.0
lambda_Ti = 1.0
lambda_Gi = 1.0

# Sample limiting values - Again, these are just placeholders and should be defined appropriately
V_lim = [1.0 for _ in Npq]
T_lim = [1.0 for _ in Nt]
QG_lim = [50 for _ in Ng]

# Apply the PSO algorithm
xopt, fopt = pso(fitness_function, lb, ub, args=(ppc,), swarmsize=50, maxiter=500)

print("Optimal solution:", xopt)
print("Objective function value:", fopt)

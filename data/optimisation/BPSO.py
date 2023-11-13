import numpy as np
import matplotlib.pyplot as plt

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
        return self.gbest_position, self.gbest_value

    def plot_convergence_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.gbest_value_history, linewidth=2)
        plt.title("BPSO Convergence Curve")
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness Value')
        plt.grid(True)
        plt.show()

# Example Objective Function
def objective_function(x):
    return sum(x)

# Parameters
n_particles = 150
dimensions = 30
max_iter = 300

# Initialize and run BPSO
bpso = BinaryPSO(n_particles, dimensions, objective_function, max_iter)
best_position, best_value = bpso.run()

print("Best Position:", best_position)
print("Best Value:", best_value)

bpso.plot_convergence_curve()
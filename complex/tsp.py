import numpy as np
import matplotlib.pyplot as plt
from random import random, seed

# Set random seed for reproducibility
seed(42)
np.random.seed(42)

# Parameters
NUM_SPECIES = 5  # Number of species (components) in the system
TIME_STEPS = 50  # Number of time steps to simulate
INTERACTION_STRENGTH = 0.1  # Strength of species interactions

# Initial population sizes (random between 10 and 100)
initial_populations = np.random.uniform(10, 100, NUM_SPECIES)

# Define interaction matrix: how species affect each other (+ for benefit, - for harm)
# Random values between -1 and 1, scaled by interaction strength
interaction_matrix = np.random.uniform(-1, 1, (NUM_SPECIES, NUM_SPECIES)) * INTERACTION_STRENGTH
np.fill_diagonal(interaction_matrix, 0)  # No self-interaction

# Function to simulate one time step of population dynamics
def update_populations(populations, matrix, problem_solving=False, target_pop=50):
    new_populations = populations.copy()
    for i in range(NUM_SPECIES):
        # Base growth rate (logistic-like)
        growth = populations[i] * (1 - populations[i] / 200) * 0.1
        
        if problem_solving:
            # Simplified problem-solving: adjust population toward a target
            if populations[i] < target_pop:  # Problem: too few individuals
                growth += (target_pop - populations[i]) * 0.2  # Boost growth
            elif populations[i] > target_pop:  # Problem: too many individuals
                growth -= (populations[i] - target_pop) * 0.2  # Reduce growth
        else:
            # Complex interactions: each species affects others via the matrix
            interaction_effect = sum(matrix[i, j] * populations[j] for j in range(NUM_SPECIES))
            growth += interaction_effect
        
        # Update population, ensuring it stays non-negative
        new_populations[i] = max(populations[i] + growth, 0)
    
    return new_populations

# Simulate two systems
evolving_system = [initial_populations.copy()]  # Tracks population over time
stagnant_system = [initial_populations.copy()]

# Run simulation
for t in range(TIME_STEPS):
    # Stagnant system: always uses complex interactions
    stagnant_next = update_populations(stagnant_system[-1], interaction_matrix, problem_solving=False)
    stagnant_system.append(stagnant_next)
    
    # Evolving system: switches to problem-solving after 20 steps
    if t < 20:
        evolving_next = update_populations(evolving_system[-1], interaction_matrix, problem_solving=False)
    else:
        evolving_next = update_populations(evolving_system[-1], interaction_matrix, problem_solving=True, target_pop=50)
    evolving_system.append(evolving_next)

# Convert lists to arrays for plotting
evolving_system = np.array(evolving_system)
stagnant_system = np.array(stagnant_system)

# Plotting
plt.figure(figsize=(12, 6))

# Evolving system
plt.subplot(1, 2, 1)
for i in range(NUM_SPECIES):
    plt.plot(evolving_system[:, i], label=f'Species {i+1}')
plt.axvline(x=20, color='gray', linestyle='--', label='Switch to Problem-Solving')
plt.title('Evolving System: Complex to Problem-Solving')
plt.xlabel('Time Step')
plt.ylabel('Population')
plt.legend()
plt.grid(True)

# Stagnant system
plt.subplot(1, 2, 2)
for i in range(NUM_SPECIES):
    plt.plot(stagnant_system[:, i], label=f'Species {i+1}')
plt.title('Stagnant System: Complex Interactions Only')
plt.xlabel('Time Step')
plt.ylabel('Population')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print a summary
print("Evolving System Summary:")
print(f"Final populations: {evolving_system[-1].round(1)}")
print("Stagnant System Summary:")
print(f"Final populations: {stagnant_system[-1].round(1)}")

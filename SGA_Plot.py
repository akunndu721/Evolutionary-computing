#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Simple genetic Algorithm
import random
import matplotlib.pyplot as plt
import numpy as np

# DeJong Evaluation Functions
def sphere(chromosome, num_bits,min_value, max_value):
    value = int(''.join(map(str, chromosome)), 2) / (2**num_bits - 1) * (max_value - min_value) + min_value
    return value**2

def step(chromosome, num_bits, min_value, max_value):
    value = int(''.join(map(str, chromosome)), 2) / (2**num_bits - 1) * (max_value - min_value) + min_value
    return abs(value)

def rosenbrock(chromosome, num_bits, min_value, max_value):
    value = int(''.join(map(str, chromosome)), 2) / (2**num_bits - 1) * (max_value - min_value) + min_value
    return (1 - value)**2

def quartic(chromosome, num_bits, min_value, max_value):
    max_value = 1.28
    min_value = -1.28
    value = int(''.join(map(str, chromosome)), 2) / (2**num_bits - 1) * (max_value - min_value) + min_value
    return value**4

def foxholes(chromosome, num_bits, min_value, max_value):
    max_value = 65.536
    min_value = -65.536
    value = int(''.join(map(str, chromosome)), 2) / (2**num_bits - 1) * (max_value - min_value) + min_value
    a = [-32, -16, 0, 16, 32]
    return sum(1 / ((value - ai)**6 + (value - bi)**6 + 1) for ai in a for bi in a)

# Step 3: Crossover and Mutation
def crossover(parent1, parent2, crossover_prob):
    if random.random() < crossover_prob:
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_prob):
    mutated_individual = list(individual)
    for i in range(len(mutated_individual)):
        if random.random() < mutation_prob:
            mutated_individual[i] = '1' if mutated_individual[i] == '0' else '0'
    return ''.join(mutated_individual)


# Genetic Algorithm
def genetic_algorithm(population_size, num_generations, crossover_prob, mutation_prob, evaluation_function):
    num_bits = 10  # Number of bits for binary encoding
    binary_precision = 2**num_bits - 1
    min_value = -5.12  # Define the lower bound for DeJong functions
    max_value = 5.12   # Define the upper bound for DeJong functions

    # Initialize population
    population = [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(population_size)]

    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness_scores.append(evaluation_function(list(individual), num_bits, min_value, max_value))
            

        # Select parents based on fitness
        parents = random.choices(population, weights=fitness_scores, k=population_size)

        # Create new generation
        new_population = []
        for i in range(0, population_size, 2):
            child1, child2 = crossover(parents[i], parents[i+1], crossover_prob)
            new_population.extend([mutate(child1, mutation_prob), mutate(child2, mutation_prob)])

        population = new_population

    # Return best solution
    best_individual = max(population, key=lambda x: evaluation_function(list(x), num_bits, min_value, max_value))
    best_fitness = evaluation_function(list(best_individual), num_bits,min_value, max_value)
    return best_individual, best_fitness


def genetic_algorithm_with_plot(population_size, num_generations, crossover_prob, mutation_prob, evaluation_function, random_seed):
    random.seed(random_seed)

    num_bits = 10
    binary_precision = 2**num_bits - 1
    min_value = -5.12
    max_value = 5.12

    population = [''.join(random.choice('01') for _ in range(num_bits)) for _ in range(population_size)]

    best_fitnesses = []

    for generation in range(num_generations):
        fitness_scores = []

        for individual in population:
            fitness_scores.append(evaluation_function(list(individual), num_bits, min_value, max_value))

        parents = random.choices(population, weights=fitness_scores, k=population_size)

        new_population = []

        for i in range(0, population_size, 2):
            child1, child2 = crossover(parents[i], parents[i+1], crossover_prob)
            new_population.extend([mutate(child1, mutation_prob), mutate(child2, mutation_prob)])

        population = new_population

        best_individual = max(population, key=lambda x: evaluation_function(list(x), num_bits, min_value, max_value))
        best_fitness = evaluation_function(list(best_individual), num_bits, min_value, max_value)
        best_fitnesses.append(best_fitness)

    return best_fitnesses


# Define a list of DeJong functions and their names
dejong_functions = [
    (sphere, "Sphere Function"),
    (step, "Step Function"),
    (rosenbrock, "Rosenbrock Function"),
    (quartic, "Quartic Function"),
    (foxholes, "Foxholes Function")
]

# Define the parameter combinations to experiment with
parameter_combinations = [
    (50, 100, 0.7, 0.001),
    (100, 200, 0.8, 0.002),
    (30, 150, 0.6, 0.001),
    (50, 75, 0.95, 0.05),
    # Add more combinations as needed
]

# Loop over each DeJong function
for dejong_function, function_name in dejong_functions:
    #print(f"Running Experiments for {function_name}\033...")
    print(f"Running Experiments for \033[1m{function_name}\033[0m...")
    
    for population_size, num_generations, crossover_prob, mutation_prob in parameter_combinations:
        print(f"Parameters: Population Size={population_size}, Generations={num_generations}, Crossover Probability={crossover_prob}, Mutation Probability={mutation_prob}")
        
        # Run the genetic algorithm
        best_solution, best_fitness = genetic_algorithm(population_size=population_size, num_generations=num_generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, evaluation_function=dejong_function)

        print(f"Best Solution for {function_name}: {best_solution}")
        print(f"Best Fitness for {function_name}: {best_fitness}\n")



# In[6]:


import random
import matplotlib.pyplot as plt
import numpy as np

# ... (previous code remains the same)

# Define the parameter combinations to experiment with
parameter_combinations = [
    (50, 100, 0.7, 0.001),
    (100, 200, 0.8, 0.002),
    (30, 150, 0.6, 0.001),
    (50, 75, 0.95, 0.05),
    # Add more combinations as needed
]

# Loop over each DeJong function
for dejong_function, function_name in dejong_functions:
    print(f"Running Experiments for \033[1m{function_name}\033[0m...")
    
    avg_min_fitness = np.zeros(100)
    avg_max_fitness = np.zeros(100)
    avg_avg_fitness = np.zeros(100)

    avg_min_obj_value = np.zeros(100)
    avg_max_obj_value = np.zeros(100)
    avg_avg_obj_value = np.zeros(100)

    for seed in range(30):
        
        # Get min, max and avg fitnesses for this seed
        min_fitnesses = np.array(genetic_algorithm_with_plot(population_size=50, num_generations=100, crossover_prob=0.7, mutation_prob=0.001, evaluation_function=dejong_function, random_seed=seed))
        max_fitnesses = np.array(genetic_algorithm_with_plot(population_size=100, num_generations=100, crossover_prob=0.7, mutation_prob=0.001, evaluation_function=dejong_function, random_seed=seed))
        avg_fitnesses = np.array(genetic_algorithm_with_plot(population_size=150, num_generations=100, crossover_prob=0.7, mutation_prob=0.001, evaluation_function=dejong_function, random_seed=seed))
        
        avg_min_fitness += min_fitnesses
        avg_max_fitness += max_fitnesses
        avg_avg_fitness += avg_fitnesses

        min_obj_values = min_fitnesses**0.5
        max_obj_values = max_fitnesses**0.5
        avg_obj_values = avg_fitnesses**0.5

        avg_min_obj_value += min_obj_values
        avg_max_obj_value += max_obj_values
        avg_avg_obj_value += avg_obj_values

    avg_min_fitness /= 30
    avg_max_fitness /= 30
    avg_avg_fitness /= 30

    avg_min_obj_value /= 30
    avg_max_obj_value /= 30
    avg_avg_obj_value /= 30

    # Plot fitness progression
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(100), avg_min_fitness, label='Min Fitness')
    plt.plot(range(100), avg_max_fitness, label='Max Fitness')
    plt.plot(range(100), avg_avg_fitness, label='Avg Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Progression - {function_name}')
    plt.legend()

    # Plot objective function values
    plt.subplot(1, 2, 2)
    plt.plot(range(100), avg_min_obj_value, label='Min Objective Value')
    plt.plot(range(100), avg_max_obj_value, label='Max Objective Value')
    plt.plot(range(100), avg_avg_obj_value, label='Avg Objective Value')
    plt.xlabel('Generation')
    plt.ylabel('Objective Value')
    plt.title(f'Objective Value Progression - {function_name}')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[97]:


# Simple Genetic Alogorithm

import random
import matplotlib.pyplot as plt

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

# Crossover and Mutation
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
            fitness_scores.append(evaluation_function(list(individual), num_bits,  min_value, max_value))
            

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
    best_fitness = evaluation_function(list(best_individual), num_bits, min_value, max_value)
    return best_individual, best_fitness


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
        


# In[98]:


# CHC Genetic Alogorithm
import random

# DeJong Evaluation Functions
def sphere(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return value**2

def step(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return abs(value)

def rosenbrock(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return (1 - value)**2

def quartic(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (1.28 - (-1.28)) + (-1.28)
    return value**4

def foxholes(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1)  * (65.536 - (-65.536)) + (-65.536)
    a = [-32, -16, 0, 16, 32]
    return sum(1 / ((value - ai)**6 + (value - bi)**6 + 1) for ai in a for bi in a)

# Crossover function (uniform crossover)
def crossover(parent1, parent2, prob):
    child1 = []
    child2 = []
    for bit1, bit2 in zip(parent1, parent2):
        if random.random() < prob:
            child1.append(bit2)
            child2.append(bit1)
        else:
            child1.append(bit1)
            child2.append(bit2)
    return ''.join(child1), ''.join(child2)

# Mutation function (bit-flip mutation)
def mutate(chromosome, mutation_rate):
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(mutated_chromosome)

# CHC Adaptive Search Algorithm
def chc_adaptive_search(chromosome_length, population_size, divergence_rate, dejong_func, crossover_prob, mutation_prob ):
    population = [''.join(random.choice('01') for _ in range(chromosome_length)) for _ in range(population_size)]
    best_individual = max(population, key=dejong_func)
    d = chromosome_length // 4
    t = 0
    best_solution = None  # Initialize best_solution to None
    best_fitness = float('inf')
    
    while t < divergence_rate:
        # Crossover
        children = [crossover(parent1, parent2, crossover_prob) for parent1, parent2 in zip(population[::2], population[1::2])]
        children = [mutate(child, mutation_prob) for child in children]  # Apply mutation
        
        # Evaluate fitness of children
        children_fitness = [dejong_func(list(map(int, child))) for child in children]
        
        # Select best individuals from parents and children
        combined_population = population + children
        new_population = sorted(combined_population, key=dejong_func, reverse=True)[:population_size]
        
        # If new population is the same as the previous population, decrement d
        if new_population == population:
            d -= 1
            if d < 0:
                population = [''.join(random.choice('01') for _ in range(chromosome_length)) for _ in range(population_size)]
                d = chromosome_length // 4
        else:
            d = chromosome_length // 4
        
        # Set the best individual from the previous generation
        new_population[0] = best_individual
        
        # Update population
        population = new_population
        
        t += 1
    
        # Track the best solution of the current generation
        current_best_individual = max(population, key=dejong_func)
        current_best_fitness = dejong_func(list(map(int, current_best_individual)))
        
        # Update the best_solution if it has the same chromosome length
        if len(current_best_individual) == chromosome_length:
            if best_solution is None or current_best_fitness > dejong_func(list(map(int, best_solution))):
                best_solution = current_best_individual
    
    # Evaluate the fitness of the best_solution
    best_fitness = dejong_func(list(map(int, best_solution))) if best_solution else None
    
    return best_solution, best_fitness

# Define a stopping condition function (you should define this based on your specific problem)
def stopping_condition_met():
    # Define your own stopping condition logic here
    return False

# Define parameters
chromosome_length = 10
divergence_rate = 100    
    
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
    ( 50, 75, 0.95, 0.05),
    # Add more combinations as needed
]

# Loop over each DeJong function
for dejong_function, function_name in dejong_functions:
    #print(f"Running Experiments for {function_name}\033...")
    print(f"Running Experiments for \033[1m{function_name}\033[0m...")
    
    for population_size, num_generations, crossover_prob, mutation_prob in parameter_combinations:
        print(f"Parameters: Population Size={population_size}, Generations={num_generations}, Crossover Probability={crossover_prob}, Mutation Probability={mutation_prob}")
        
        # Run the CHC genetic algorithm
 
        best_solution, best_fitness = chc_adaptive_search(chromosome_length, population_size, divergence_rate, dejong_function, crossover_prob, mutation_prob )

        print(f"Best Solution for {function_name}: {best_solution}")
        print(f"Best Fitness for {function_name}: {best_fitness}\n")
        


# In[99]:


import random
import matplotlib.pyplot as plt

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
    (50, 100, 0.7, 0.001, 10, 100),
    (100, 200, 0.8, 0.002, 10, 100),
    (30, 150, 0.6, 0.001, 10, 100),
    (50, 75, 0.95, 0.05, 10, 100),
    # Add more combinations as needed
]

# Create a dictionary to store results
results = {}

# Loop over each DeJong function
for dejong_function, function_name in dejong_functions:
    results[function_name] = {}

    for population_size, num_generations, crossover_prob, mutation_prob, chromosome_length, divergence_rate in parameter_combinations:
        #print(f"Parameters: Population Size={population_size}, Generations={num_generations}, Crossover Probability={crossover_prob}, Mutation Probability={mutation_prob}")
        avg_evaluations_sga = 0
        avg_evaluations_chc_ga = 0
        
        # Run SGA and CHC-GA multiple times
        num_runs = 10  # Adjust as needed
        for _ in range(num_runs):
            
            # Run SGA
            best_solution_sga, best_fitness_sga = genetic_algorithm(population_size=population_size, num_generations=num_generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, evaluation_function=dejong_function)
            avg_evaluations_sga += best_fitness_sga           
            
            # Run CHC-GA
            #best_solution_chc_ga, best_fitness_chc_ga = chc_adaptive_search(chromosome_length=chromosome_length, population_size=population_size, divergence_rate=divergence_rate, dejong_func=dejong_function, crossover_prob=crossover_prob, mutation_prob=mutation_prob )
            #avg_evaluations_chc_ga += best_fitness_chc_ga
        
        # Calculate average for SGA and CHC-GA
        avg_evaluations_sga /= num_runs
        avg_evaluations_chc_ga /= num_runs
        
        # Store results
        #results[function_name][str(parameters)] = (avg_evaluations_sga)
        results[function_name][(population_size, num_generations, crossover_prob, mutation_prob)] = avg_evaluations_sga

# Print the results in a table format
table = PrettyTable()
table.field_names = ["DeJong Function", "Population Size", "Generations", "Crossover Prob", "Mutation Prob", "Avg Func Evals (SGA)"]

for function_name, parameter_results in results.items():
    for parameters, avg_evaluations_sga in parameter_results.items():
        table.add_row([function_name] + list(parameters) + [f"{avg_evaluations_sga:.2f}"])

print(table)


# In[100]:


import random
from prettytable import PrettyTable

# DeJong Evaluation Functions
def sphere(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return value**2

def step(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return abs(value)

def rosenbrock(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (5.12 - (-5.12)) + (-5.12)
    return (1 - value)**2

def quartic(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1) * (1.28 - (-1.28)) + (-1.28)
    return value**4

def foxholes(chromosome):
    value = int(''.join(map(str, chromosome)), 2) / (2**len(chromosome) - 1)  * (65.536 - (-65.536)) + (-65.536)
    a = [-32, -16, 0, 16, 32]
    return sum(1 / ((value - ai)**6 + (value - bi)**6 + 1) for ai in a for bi in a)


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
    (50, 100, 0.7, 0.001, 10, 100),
    (100, 200, 0.8, 0.002, 10, 100),
    (30, 150, 0.6, 0.001, 10, 100),
    (50, 75, 0.95, 0.05, 10, 100),
    # Add more combinations as needed
]

# Create a dictionary to store results
results = {}

# Loop over each DeJong function
for dejong_function, function_name in dejong_functions:
    results[function_name] = {}

    for population_size, num_generations, crossover_prob, mutation_prob, chromosome_length, divergence_rate in parameter_combinations:
        #print(f"Parameters: Population Size={population_size}, Generations={num_generations}, Crossover Probability={crossover_prob}, Mutation Probability={mutation_prob}")
        avg_evaluations_sga = 0.0
        avg_evaluations_chc_ga = 0.0
        
        # Run SGA and CHC-GA multiple times
        num_runs = 10  # Adjust as needed
        for _ in range(num_runs):
            
            # Run SGA
            #best_solution_sga, best_fitness_sga = genetic_algorithm(population_size=population_size, num_generations=num_generations, crossover_prob=crossover_prob, mutation_prob=mutation_prob, evaluation_function=dejong_function)
            #avg_evaluations_sga += best_fitness_sga           
            
            # Run CHC-GA
            best_solution_chc_ga, best_fitness_chc_ga = chc_adaptive_search(chromosome_length=chromosome_length, population_size=population_size, divergence_rate=divergence_rate, dejong_func=dejong_function, crossover_prob=crossover_prob, mutation_prob=mutation_prob )
            if best_fitness_chc_ga is not None:
                avg_evaluations_chc_ga += best_fitness_chc_ga
        
        # Calculate average for SGA and CHC-GA
        avg_evaluations_sga /= num_runs
        avg_evaluations_chc_ga /= num_runs
        
        # Store results
        results[function_name][(population_size, num_generations, crossover_prob, mutation_prob)] = (avg_evaluations_chc_ga)

# Print the results in a table format
table = PrettyTable()
table.field_names = ["DeJong Function", "Population Size", "Generations", "Crossover Prob", "Mutation Prob", "Avg Func Evals (CHC-GA)"]

for function_name, parameter_results in results.items():
    for parameters, avg_evaluations_chc_ga in parameter_results.items():
        table.add_row([function_name] + list(parameters) + [f"{avg_evaluations_chc_ga:.2f}"])

print(table)


# In[ ]:





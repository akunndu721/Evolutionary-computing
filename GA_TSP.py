#!/usr/bin/env python
# coding: utf-8

# # **The Traveling Sales Problem using genetic algorithm**
# The problem is to minimize the distance travelled by a salesperson as they visit 
# all the cities, visiting each city exactly once. 

# In[9]:


import random
import math
import re
from prettytable import PrettyTable

#This class to define the stucture of node with city id and its 2D coordinates (x,y)
class Node:
    def __init__(self, id, x, y):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)

#Input city data and its co-ordinates (x, y) - This is various benchmark data
#file_name = "burma14.tsp"
#file_name = "eil51.tsp"
#file_name = "berlin52.tsp"  
file_name = "eil76.tsp" 
#file_name = "lin105.tsp"
#file_name = "bier127.tsp"
#file_name = "gr137.tsp"
#file_name = "rat195.tsp"
#file_name = "lin318.tsp" 
#file_name = "rat575.tsp"


#Creating data set based on input city files
dataset = []
with open(file_name, "r") as f:
    for line in f:
        new_line = re.split(r'\s+', line.strip())
        if new_line[0].isdigit():
            id, x, y = new_line[0], float(new_line[1]), float(new_line[2])
            dataset.append(Node(id=id, x=x, y=y))
        
N = len(dataset) # Number of cities
print("Number of Cities:" + str(N))

#Creating distance matrix using Euclidean distance formula
def distance_matrix(node_list):
    matrix = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(0, len(matrix)-1):
        for j in range(0, len(matrix[0])-1):
            matrix[node_list[i].id][node_list[j].id] = math.sqrt(
                pow((node_list[i].x - node_list[j].x), 2) + pow((node_list[i].y - node_list[j].y), 2)
            )   
    return matrix

matrix = distance_matrix(dataset)

# The class is for Chromosome for maintaing Node list. This will be used in crossover, mutation operations etc.
class Chromosome:
    def __init__(self, node_list):
        self.chromosome = node_list

        chr_representation = []
        for i in range(0, len(node_list)):
            chr_representation.append(self.chromosome[i].id)
        self.chr_representation = chr_representation
        
        distance = 0
        for j in range(1, len(self.chr_representation) - 1):  # get distances from the matrix
            distance += matrix[self.chr_representation[j]-1][self.chr_representation[j + 1]-1]
        self.cost = distance

        self.fitness_value = 1 / self.cost


# create a random chromosome and start and end points should be same
def create_random_list(n_list):
    
    start = n_list[0]  
    temp = n_list[1:]
    temp = random.sample(temp, len(temp))  
    temp.insert(0, start)  
    temp.append(start) 
    return temp


# initialize the population
def initialization(data, pop_size):
    initial_population = []
    for i in range(0, pop_size):  
        temp = create_random_list(data)
        new_ch = Chromosome(temp)
        initial_population.append(new_ch)
    return initial_population


# Select parent chromosomes to create child chromosomes using tournament selection
def selection(population):  
    ticket_1, ticket_2, ticket_3, ticket_4 = random.sample(range(0, 99), 4) 

    # create candidate chromosomes based on ticket numbers
    candidate_1 = population[ticket_1]
    candidate_2 = population[ticket_2]
    candidate_3 = population[ticket_3]
    candidate_4 = population[ticket_4]

    # select the winner according to their costs
    if candidate_1.fitness_value > candidate_2.fitness_value:
        winner = candidate_1
    else:
        winner = candidate_2

    if candidate_3.fitness_value > winner.fitness_value:
        winner = candidate_3

    if candidate_4.fitness_value > winner.fitness_value:
        winner = candidate_4

    return winner  


# Two points crossover
def crossover(p_1, p_2):
    point_1, point_2 = random.sample(range(1, len(p_1.chromosome)-1), 2)
    begin = min(point_1, point_2)
    end = max(point_1, point_2)

    child_1_1 = p_1.chromosome[:begin]
    child_1_2 = p_1.chromosome[end:]
    child_1 = child_1_1 + child_1_2
    child_2 = p_2.chromosome[begin:end+1]

    child_1_remain = [item for item in p_2.chromosome[1:-1] if item not in child_1]
    child_2_remain = [item for item in p_1.chromosome[1:-1] if item not in child_2]

    child_1 = child_1_1 + child_1_remain + child_1_2
    child_2 += child_2_remain

    child_2.insert(0, p_2.chromosome[0])
    child_2.append(p_2.chromosome[0])

    return child_1, child_2


# Mutation operation
def mutation(chromosome):  # swap two nodes of the chromosome
    mutation_index_1, mutation_index_2 = random.sample(range(1, 10), 2)
    chromosome[mutation_index_1], chromosome[mutation_index_2] = chromosome[mutation_index_2], chromosome[mutation_index_1]
    return chromosome


# Find the best chromosome of the generation based on the cost
def find_best(generation):
    best = generation[0]
    for n in range(1, len(generation)):
        if generation[n].cost < best.cost:
            best = generation[n]
    return best

# Use elitism, crossover, mutation operators to create a new generation based on a previous generation

def create_new_generation(previous_generation, crossover_probability, mutation_rate):
    new_generation = [find_best(previous_generation)]  # This is for elitism. Keep the best of the previous generation.

    # Use two chromosomes and create two chromosomes. So, iteration size will be half of the population size!
    for a in range(0, int(len(previous_generation)/2)):
        parent_1 = selection(previous_generation)
        parent_2 = selection(previous_generation)

        if random.random() < crossover_probability:
            child_1, child_2 = crossover(parent_1, parent_2)  # This will create node lists, we need Chromosome objects
            child_1 = Chromosome(child_1)
            child_2 = Chromosome(child_2)
        else:
            child_1 = parent_1
            child_2 = parent_2

        if random.random() < mutation_rate:
            mutated = mutation(child_1.chromosome)
            child_1 = Chromosome(mutated)

        new_generation.append(child_1)
        new_generation.append(child_2)

    return new_generation


def genetic_algorithm(num_of_generations, pop_size, cross_prob, mutation_rate, data_list):
    
    new_gen = initialization(data_list, pop_size) 

    costs_for_plot = []  
    for iteration in range(0, num_of_generations):
        new_gen = create_new_generation(new_gen, cross_prob, mutation_rate)  
        costs_for_plot.append(find_best(new_gen).cost)

    return new_gen, costs_for_plot


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

def draw_path(solution):
    x_list = []
    y_list = []

    for m in range(0, len(solution.chromosome)):
        x_list.append(solution.chromosome[m].x)
        y_list.append(solution.chromosome[m].y)

    fig, ax = plt.subplots()
    plt.scatter(x_list, y_list)  # alpha=0.5

    ax.plot(x_list, y_list, '--', lw=2, color='black', ms=10)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 1300)

    plt.show()
    
def draw_cost_generation(y_list, generation, pop_size, cross_prob, mutation_rate):
    x_list = np.arange(1, len(y_list) + 1)  # create a numpy list from 1 to the numbers of generations

    plt.plot(x_list, y_list)

    plt.title("Tour Cost through Generations")
    plt.xlabel("Generations")
    plt.ylabel("Cost")

    # Add annotation with parameter values
    parameter_label = f'Generation: {generation}\nPopulation Size: {pop_size}\nCrossover Probability: {cross_prob}\nMutation Rate: {mutation_rate}'
    plt.annotate(parameter_label, xy=(0.5, 0.85), xycoords='axes fraction', fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.show()


# In[11]:


import random

# Set the number of runs
num_runs = 30

# Genetic Algorithm Parameter Combinations
parameter_combinations = [
    (700, 400, 0.9, 0.8),
    # Add more combinations as needed
]

# Loop through each parameter combination
for params in parameter_combinations:
    # Extract parameters
    numbers_of_generations, population_size, crossover_probability, mutation_rate = params

    # Loop through multiple runs with different random seeds
    for run in range(num_runs):
        # Set a different random seed for each run
        random_seed = run + 1
        random.seed(random_seed)

        # Run genetic algorithm
        last_generation, y_axis = genetic_algorithm(
            num_of_generations=numbers_of_generations,
            pop_size=population_size,
            cross_prob=crossover_probability,
            mutation_rate=mutation_rate,
            data_list=dataset
        )

        # Find the best solution
        best_solution = find_best(last_generation)

        # Display results for each run
        best_cost_last_generation = last_generation[0].cost
        best_path_last_generation = last_generation[0].chr_representation
        print(f"Run {run + 1} - Minimum tour length: {best_cost_last_generation:.2f}")
        print(f"Run {run + 1} - Best path: {best_path_last_generation}")

        # Draw cost vs generation plot for each run
        draw_cost_generation(y_axis, numbers_of_generations, population_size, crossover_probability, mutation_rate)


# In[12]:


import matplotlib.pyplot as plt
import re

def read_coordinates_from_file(file_name):
    coordinates = []
    with open(file_name, 'r') as file:
        for line in file:
            new_line = re.split(r'\s+', line.strip())
            if new_line[0].isdigit():
                id, x, y = new_line[0], float(new_line[1]), float(new_line[2])
                coordinates.append((x, y))
    return coordinates

# Read coordinates from the file
coordinates = read_coordinates_from_file(file_name)

# Extract x and y coordinates from the list of coordinates
x_coords, y_coords = zip(*coordinates)

# Plot the coordinates
plt.figure(figsize=(8, 8))
plt.scatter(x_coords, y_coords, c='red', marker='o', label='City')

# Annotate each point with its index
for i, (x, y) in enumerate(coordinates):
    plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

# Connect the points in the order of the best path indices
for start, end in best_path_indices:
    x_start, y_start = coordinates[start]
    x_end, y_end = coordinates[end]
    plt.plot([x_start, x_end], [y_start, y_end], linestyle='-', color='blue')

# Connect back to the starting point
x_start, y_start = coordinates[best_path_indices[-1][1]]
x_end, y_end = coordinates[best_path_indices[0][0]]
plt.plot([x_start, x_end], [y_start, y_end], linestyle='-', color='blue')

# Mark the starting point with a different marker or color
start_index = best_path_indices[0][0]
x_start, y_start = coordinates[start_index]
plt.scatter(x_start, y_start, c='green', marker='s', s=100, label='Start City')

# Show the legend
plt.legend()

# Show the plot
plt.title(f'Coordinates Plot with Best Path\n\nDataset Instance={file_name}\nBest Path length:={best_cost_last_generation:.2f}')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


# In[ ]:





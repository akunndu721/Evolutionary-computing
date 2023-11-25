#!/usr/bin/env python
# coding: utf-8

# ## **Traveling Salesman Problem Solutions with Ant Colony Optimization and Genetic Algorithms.**

# In[2]:


import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, decay, alpha=1, beta=1):
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self, n):
        self.all_inds = range(len(distances))
        self.shortest_path_ = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(n):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.pheromone, self.all_inds, self.distances)

            self.pheromone *= self.decay

            self.shortest_path_ = min(all_paths, key=lambda x: x[1])
            if self.shortest_path_[1] < all_time_shortest_path[1]:
                all_time_shortest_path = self.shortest_path_                
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, pheromone, all_inds, distances):
        pheromone *= self.decay
        for path, dist in all_paths:
            for move in path:
                pheromone[move] += 1.0 / distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path_dist = 0
            path = []
            visited_inds = set()
            visited_inds.add(self.all_inds[i])
            prev = self.all_inds[i]
            for j in range(len(self.distances) - 1):
                move = self.pick_move(self.pheromone[prev], self.distances[prev], visited_inds)
                path_dist += self.distances[prev][move]
                path.append((prev, move))
                prev = move
                visited_inds.add(move)
            all_paths.append((path, path_dist))
        return all_paths
    
    def pick_move(self, pheromone, dist, visited_inds):
        pheromone = np.copy(pheromone)
        pheromone[list(visited_inds)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        # Normalize row only if it's not empty
        if row.sum() > 0:
            norm_row = row / row.sum()
            move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        else:
            # If the row is empty, choose a random move
            move = np.random.choice(list(set(self.all_inds) - visited_inds), 1)[0]

        return move


class GeneticAlgorithm:
    def __init__(self, population_size, elite_size, mutation_rate):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate

    def crossover(self, parent1, parent2):
        # Order Crossover (OX1) - a common crossover method for TSP
        start, end = sorted(np.random.choice(len(parent1), 2, replace=False))
        child = [-1] * len(parent1)
        child[start:end] = parent1[start:end]
        remaining = [item for item in parent2 if item not in child]
        index = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = remaining[index]
                index += 1
        return child

    def mutate(self, individual):
        # Swap Mutation - swap two cities in the individual
        mutate_index1, mutate_index2 = np.random.choice(len(individual), 2, replace=False)
        individual[mutate_index1], individual[mutate_index2] = individual[mutate_index2], individual[mutate_index1]
        return individual

    def evolve(self, population):
        elite_size = int(self.elite_size * len(population))
        elites = sorted(population, key=lambda x: x[1])[:elite_size]

        # Crossover
        children = []
        while len(children) < (len(population) - elite_size):
            parent1, parent2 = np.random.choice(elites, 2, replace=False)
            child = self.crossover(parent1[0], parent2[0])
            children.append((child, -1))

        # Mutation
        for i in range(len(children)):
            if np.random.rand() < self.mutation_rate:
                children[i] = (self.mutate(children[i][0]), -1)

        # Combine elites and children
        population = elites + children
        return population

class ACOGA:
    def __init__(self, distances, n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, generations):
        self.aco = AntColony(distances, n_ants, decay, alpha, beta)
        self.ga = GeneticAlgorithm(ga_population_size, ga_elite_size, ga_mutation_rate)
        self.generations = generations

    def run(self):
        for gen in range(self.generations):
            aco_shortest_path = self.aco.run(1)
            ga_population = self.generate_ga_population(aco_shortest_path)
            ga_population = self.ga.evolve(ga_population)
            best_ga_solution = min(ga_population, key=lambda x: x[1])
            self.aco.pheromone *= self.aco.decay
            self.aco.spread_pheronome([best_ga_solution], self.aco.pheromone, self.aco.all_inds, self.aco.distances)

        return aco_shortest_path

    def generate_ga_population(self, aco_solution):
        # Generate GA population from ACO solution
        ga_population = [(aco_solution[0], -1) for _ in range(self.ga.population_size)]
        return ga_population

# Example usage:
if __name__ == "__main__":
    distances = np.array([[np.inf, 2, 2, 5, 7],
                         [2, np.inf, 4, 8, 2],
                         [2, 4, np.inf, 1, 3],
                         [5, 8, 1, np.inf, 2],
                         [7, 2, 3, 2, np.inf]])

    aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=10, ga_elite_size=2, ga_mutation_rate=0.2, generations=20)
    result = aco_ga.run()

    print("ACO-GA Combined Solution:", result)


# In[3]:


import numpy as np
import re

def euclidean_distance(coord1, coord2):
    return np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)

def read_coordinates_from_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            new_line = re.split(r'\s+', line.strip())
            if new_line[0].isdigit():
                id, x, y = new_line[0], float(new_line[1]), float(new_line[2])
                coordinates.append((x, y))
    return coordinates

# Specify the path to your file containing coordinates
file_path = "burma14.tsp"
#file_path = "berlin52.tsp"

# Read coordinates from the file
coordinates = read_coordinates_from_file(file_path)

# Create a distance matrix using NumPy array
num_points = len(coordinates)
distance_matrix = np.zeros((num_points, num_points))

for i in range(num_points):
    for j in range(num_points):
        if i != j:
            distance_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])
        else:
            distance_matrix[i, j] = np.inf

# Replace inf values with np.inf
distance_matrix = np.where(np.isinf(distance_matrix), np.inf, distance_matrix)

# Round the values in the distance array to 2 decimal places
distances = np.round(distance_matrix, 2)

print("Distance Matrix:")

# Print column indices
print("      ", end="")
for i in range(num_points):
    print(f"  {i:<6}", end="")
print()

# Print the matrix with row indices
for i in range(num_points):
    print(f"{i:2} |", end="")
    for value in distances[i]:
        print(f"{value:7.2f}", end=" ")
    print()

# Now you can proceed with your ACOGA implementation using the read coordinates.
aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=10, ga_elite_size=2, ga_mutation_rate=0.2, generations=20)
result = aco_ga.run()

best_path_indices = result[0]

print("Best Path Indices:", best_path_indices)
print("Best Path length:", np.round(result[1], 2))


# In[4]:


import matplotlib.pyplot as plt

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
plt.title('Coordinates Plot with Best Path')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


# In[5]:


# Initialize ACOGA parameters

import time

n_ants = 5
decay = 0.95
alpha = 1
beta = 2
ga_population_size = 20
ga_elite_size = 2
ga_mutation_rate = 0.2
generations = 130

# Initialize ACOGA instance
aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
               ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
               ga_mutation_rate=ga_mutation_rate, generations=generations)

# Measure execution time
start_time = time.time()

# Run ACOGA
result = aco_ga.run()

# Measure execution time
execution_time = time.time() - start_time
print("Execution Time:", execution_time, "seconds")

# Extract result data
best_path_indices = result[0]
best_path_length = np.round(result[1], 2)

# Print results
print("Best Path Indices:", best_path_indices)
print("Best Path Length:", best_path_length)


# In[6]:


# Lists to store data for plotting
generations_list = []
execution_times_list = []

# Run ACOGA for multiple generations
for generations in range(10, 501, 50):
    # Initialize ACOGA instance
    aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
                   ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
                   ga_mutation_rate=ga_mutation_rate, generations=generations)

    # Measure execution time
    start_time = time.time()

    # Run ACOGA
    result = aco_ga.run()

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Generations: {generations}, Execution Time: {execution_time:.4f} seconds")

    # Store data for plotting
    generations_list.append(generations)
    execution_times_list.append(execution_time)

    # Extract result data
    best_path_indices = result[0]
    best_path_length = np.round(result[1], 2)

    # Print results
    print("Best Path Indices:", best_path_indices)
    print("Best Path Length:", best_path_length)

# Plot time vs. generation
plt.plot(generations_list, execution_times_list, marker='o')
plt.title('Execution Time vs. Generation')
plt.xlabel('Generation')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.show()


# In[8]:


# Lists to store data for plotting
generations_list = []
execution_times_list = []
best_path_lengths_list = []

# Run ACOGA for multiple generations
for generations in range(10, 1000, 50):
    # Initialize ACOGA instance
    aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
                   ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
                   ga_mutation_rate=ga_mutation_rate, generations=generations)

    # Measure execution time
    start_time = time.time()

    # Run ACOGA
    result = aco_ga.run()

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Generations: {generations}, Execution Time: {execution_time:.4f} seconds")

    # Store data for plotting
    generations_list.append(generations)
    execution_times_list.append(execution_time)
    best_path_lengths_list.append(result[1])  # Appending the best path length

    # Extract result data
    best_path_indices = result[0]
    best_path_length = np.round(result[1], 2)

    # Print results
    print("Best Path Indices:", best_path_indices)
    print("Best Path Length:", best_path_length)

# Plot best path length vs. generation
plt.plot(generations_list, best_path_lengths_list, marker='o')
plt.title('Best Path Length vs. Generation')
plt.xlabel('Generation')
plt.ylabel('Best Path Length')
plt.grid(True)
plt.show()


# In[9]:


# Lists to store data for plotting
generations_list = []
execution_times_list = []
average_path_lengths_list = []
max_path_lengths_list = []

# Run ACOGA for multiple generations
for generations in range(10, 501, 20):
    # Initialize ACOGA instance
    aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
                   ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
                   ga_mutation_rate=ga_mutation_rate, generations=generations)

    # Measure execution time
    start_time = time.time()

    # Run ACOGA
    result = aco_ga.run()

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Generations: {generations}, Execution Time: {execution_time:.4f} seconds")

    # Store data for plotting
    generations_list.append(generations)
    execution_times_list.append(execution_time)

    # Extract result data
    best_path_indices = result[0]
    best_path_length = np.round(result[1], 2)

    # Calculate average and max path lengths
    average_path_length = np.mean([distances[i, j] for i, j in best_path_indices])
    max_path_length = np.max([distances[i, j] for i, j in best_path_indices])

    average_path_lengths_list.append(average_path_length)
    max_path_lengths_list.append(max_path_length)

    # Print results
    print("Best Path Indices:", best_path_indices)
    print("Best Path Length:", best_path_length)
    print("Average Path Length:", average_path_length)
    print("Max Path Length:", max_path_length)

# Plot average path length vs. generation
plt.plot(generations_list, average_path_lengths_list, label='Average Path Length', marker='o')

# Plot max path length vs. generation
plt.plot(generations_list, max_path_lengths_list, label='Max Path Length', marker='o')

plt.title('Path Lengths vs. Generation')
plt.xlabel('Generation')
plt.ylabel('Path Length')
plt.legend()
plt.grid(True)
plt.show()


# In[10]:


# Lists to store data for plotting
generations_list = []
execution_times_list = []
best_path_lengths_list = []

# Various Parameter combination to achieve best solution
parameter_combinations = [
    (5, 0.9, 1, 1, 10, 2, 0.1, 50),
    (7, 0.95, 2, 2, 20, 2, 0.2, 100),
    (10, 0.99, 3, 3, 30, 2, 0.2, 150),
]

for params in parameter_combinations:
    n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, generations = params
    aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
                   ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
                   ga_mutation_rate=ga_mutation_rate, generations=generations)
    
    # Measure execution time
    start_time = time.time()

    # Run ACOGA
    result = aco_ga.run()

    # Measure execution time
    execution_time = time.time() - start_time
    print(f"Generations: {generations}, Execution Time: {execution_time:.4f} seconds")

    # Store data for plotting
    generations_list.append(generations)
    execution_times_list.append(execution_time)
    best_path_lengths_list.append(result[1])  # Appending the best path length

    # Extract result data
    best_path_indices = result[0]
    best_path_length = np.round(result[1], 2)

    # Print results
    print("Best Path Indices:", best_path_indices)
    print("Best Path Length:", best_path_length)
    
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
    #plt.title('Coordinates Plot with Best Path')
    plt.title(f'Coordinates Plot with Best Path\n\nParameters:n_ants={n_ants},decay={decay}, alpha={alpha}, beta={beta}, ga_population_size={ga_population_size}, ga_elite_size={ga_elite_size}, ga_mutation_rate={ga_mutation_rate}, generations={generations}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


# In[ ]:





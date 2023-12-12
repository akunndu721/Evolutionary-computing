#!/usr/bin/env python
# coding: utf-8

# ## **Traveling Salesman Problem Solutions with Ant Colony Optimization and Genetic Algorithms.**

# In[1]:


import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, decay, alpha, beta):
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
    def __init__(self, population_size, elite_size, mutation_rate, crossover_prob):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_prob = crossover_prob  # Add crossover probability

    def crossover(self, parent1, parent2):
        # Order Crossover (OX1) - a common crossover method for TSP
        if np.random.rand() < self.crossover_prob:
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
        else:
            return parent1  # If no crossover, return the first parent as the child

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
    def __init__(self, distances, n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, ga_crossover_prob, generations):
        self.aco = AntColony(distances, n_ants, decay, alpha, beta)
        self.ga = GeneticAlgorithm(ga_population_size, ga_elite_size, ga_mutation_rate, ga_crossover_prob)
        self.generations = generations

    def run(self):
        for gen in range(self.generations):
            aco_shortest_path = self.aco.run(10)
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

    aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=10, ga_elite_size=2, ga_mutation_rate=0.2, ga_crossover_prob=0.8, generations=20)
    result = aco_ga.run()

    print("ACO-GA Combined Solution:", result)


# In[2]:


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

# Specify the path to the file containing coordinates
#file_path = "burma14.tsp"
#file_path = "eil51.tsp"
#file_path = "berlin52.tsp"
file_path = "eil76.tsp"
#file_path = "lin105.tsp"
#file_path = "bier127.tsp"
#file_path = "gr137.tsp"
#file_path = "rat195.tsp"
#file_path = "lin318.tsp"
#file_path = "rat575.tsp"

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

# ACOGA implementation using the read coordinates.
aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=400, ga_elite_size=2, ga_mutation_rate=0.8, ga_crossover_prob=0.9, generations=1000)
result = aco_ga.run()

best_path_indices = result[0]

print("Best Path Indices:", best_path_indices)
print("Best Path length:", np.round(result[1], 2))


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

# Specify the path to the file containing coordinates
#file_path = "burma14.tsp"
#file_path = "eil51.tsp"
#file_path = "berlin52.tsp"
file_path = "eil76.tsp"
#file_path = "lin105.tsp"
#file_path = "bier127.tsp"
#file_path = "gr137.tsp"
#file_path = "rat195.tsp"
#file_path = "lin318.tsp"
#file_path = "rat575.tsp"

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

# Run ACOGA with 30 different random seeds
for run in range(30):
    # Set a different random seed for each run
    random_seed = run + 1
    np.random.seed(random_seed)

    # ACOGA implementation using the read coordinates.
    aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=400, ga_elite_size=2, ga_mutation_rate=0.8, ga_crossover_prob=0.9, generations=1000)
    result = aco_ga.run()

    best_path_indices = result[0]

    print(f"\nRun {run + 1} - Best Path Indices: {best_path_indices}")
    print(f"Run {run + 1} - Best Path Length: {np.round(result[1], 2)}")


# ## **Calculating Best Path and Plot**

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
plt.title(f'Coordinates Plot with Best Path\n\nDataset Instance={file_path}\nBest Path length:={np.round(result[1], 2)}')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()


# ## **Plotting Generation vs Execution time and Generation vs Best Path Length**

# In[ ]:


import time

# Lists to store data for plotting
generations_list = []
execution_times_list = []
best_path_lengths_list = []

# Run ACOGA for multiple generations
for generations in range(10, 500, 50):
    # Initialize ACOGA instance
    aco_ga = ACOGA(distances, n_ants=5, decay=0.95, alpha=1, beta=2, ga_population_size=400, ga_elite_size=2, ga_mutation_rate=0.8, ga_crossover_prob=0.9, generations=generations)

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


# Plot time vs. generation
plt.plot(generations_list, execution_times_list, marker='o')
plt.title('Execution Time vs. Generation')
plt.xlabel('Generation')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.show()

# Plot best path length vs. generation
plt.plot(generations_list, best_path_lengths_list, marker='o')
plt.title('Best Path Length vs. Generation')
plt.xlabel('Generation')
plt.ylabel('Best Path Length')
plt.grid(True)
plt.show()


# ## **Plotting Best Path on different Parameters**

# In[ ]:


# Lists to store data for plotting
generations_list = []
execution_times_list = []
best_path_lengths_list = []

# Various Parameter combination to achieve best solution
parameter_combinations = [
    (5, 0.9, 1, 1, 100, 2, 0.7, 0.8, 500),
    (7, 0.95, 2, 2, 200, 2, 0.8, 0.7, 600),
    (10, 0.99, 3, 3, 300, 2, 0.9, 0.9, 700),
]

for params in parameter_combinations:
    n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, ga_crossover_prob, generations = params
    aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta,
                   ga_population_size=ga_population_size, ga_elite_size=ga_elite_size,
                   ga_mutation_rate=ga_mutation_rate, ga_crossover_prob = ga_crossover_prob, generations=generations)
    
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


# ## **Plotting avg-avg and max-avg versus number of generations**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Lists to store data for plotting
generations_list = []
execution_times_list = []
overall_avg_best_path_lengths_list = []
overall_max_best_path_lengths_list = []

# Define the parameter combinations for ACOGA
parameter_combinations = [(10, 5, 0.95, 1, 2, 400, 2, 0.8, 0.9)]

# Run ACOGA for multiple generations
for generations in range(10, 100, 10):
    # Initialize arrays to store path length values
    num_generations_acoga = generations
    avg_best_path_length = np.zeros((num_generations_acoga,))
    max_best_path_length = np.zeros((num_generations_acoga,))

    num_runs = 5
    for params_idx, params in enumerate(parameter_combinations):
        num_of_generations, n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, ga_crossover_prob = params

        for run in range(num_runs):
            # Initialize ACOGA instance
            aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta, ga_population_size=ga_population_size,
                           ga_elite_size=ga_elite_size, ga_mutation_rate=ga_mutation_rate, ga_crossover_prob=ga_crossover_prob,
                           generations=num_of_generations)

            # Measure execution time
            start_time = time.time()

            # Run ACOGA
            result = aco_ga.run()

            # Measure execution time
            execution_time = time.time() - start_time

            # Store data for plotting

            execution_times_list.append(execution_time)

            # Extract result data
            best_path_indices = result[0]

            # Calculate best path length
            best_path_length = np.sum([distances[i, j] for i, j in best_path_indices])

            # Update best path length arrays
            avg_best_path_length[:len(best_path_indices)] += np.array([best_path_length]) / num_runs
            max_best_path_length[:len(best_path_indices)] = np.maximum(max_best_path_length[:len(best_path_indices)], np.array([best_path_length]))

            # Print results
            #print("Best Path Indices:", best_path_indices)
           # print("Best Path Length:", best_path_length)
            
    # Print average and maximum best path lengths after all runs for each generation
    

    overall_avg_best_path_length = np.mean(avg_best_path_length[:num_generations_acoga])
    overall_max_best_path_length = np.max(max_best_path_length[:num_generations_acoga])
    
    print("generations:", generations)
    print("overall_avg_best_path_length:", overall_avg_best_path_length)
    print("overall_max_best_path_length:", overall_max_best_path_length)
    
    generations_list.append(generations)
    overall_avg_best_path_lengths_list.append(overall_avg_best_path_length)
    overall_max_best_path_lengths_list.append(overall_max_best_path_length)

# Plotting with magnified figure size
plt.figure(figsize=(12, 8))

# Plotting
plt.plot(generations_list, overall_avg_best_path_lengths_list, label='Average Best Path Length')
plt.plot(generations_list, overall_max_best_path_lengths_list, label='Max Best Path Length')

# Customize plot appearance
plt.xlabel('Generation')
plt.ylabel('Best Path Length')
plt.title('Overall Best Path Lengths vs. Generation')
plt.title(f'Tour Length vs. Generation\n\nDataset Instance={file_path}\nBest Path length:={np.round(result[1], 2)}')

plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import time

# Lists to store data for plotting
generations_list = []
execution_times_list = []
overall_avg_best_path_lengths_list = []
overall_max_best_path_lengths_list = []

# Define the parameter combinations for ACOGA

# Define the parameter combinations for ACOGA
parameter_combinations = [(10, 5, 0.95, 1, 2, 400, 2, 0.8, 0.9)]

# Run ACOGA for multiple generations
for generations in range(10, 500, 10):
    # Initialize arrays to store path length values
    num_generations_acoga = generations
    avg_best_path_length = np.zeros((num_generations_acoga,))
    max_best_path_length = np.zeros((num_generations_acoga,))

    for params_idx, params in enumerate(parameter_combinations):
        num_of_generations, n_ants, decay, alpha, beta, ga_population_size, ga_elite_size, ga_mutation_rate, ga_crossover_prob = params
        
        num_runs = 30
        for run in range(num_runs):
            # Initialize ACOGA instance
            aco_ga = ACOGA(distances, n_ants=n_ants, decay=decay, alpha=alpha, beta=beta, ga_population_size=ga_population_size,
                           ga_elite_size=ga_elite_size, ga_mutation_rate=ga_mutation_rate, ga_crossover_prob=ga_crossover_prob,
                           generations=num_of_generations)

            # Measure execution time
            start_time = time.time()

            # Run ACOGA
            result = aco_ga.run()

            # Measure execution time
            execution_time = time.time() - start_time

            # Store data for plotting

            execution_times_list.append(execution_time)

            # Extract result data
            best_path_indices = result[0]

            # Calculate best path length
            best_path_length = np.sum([distances[i, j] for i, j in best_path_indices])

            # Update best path length arrays
            avg_best_path_length[:len(best_path_indices)] = np.array([best_path_length]) 
            max_best_path_length[:len(best_path_indices)] = np.maximum(max_best_path_length[:len(best_path_indices)], np.array([best_path_length]))

            # Print results
            #print("Best Path Indices:", best_path_indices)
            #print("Best Path Length:", best_path_length)
            

    # Print average and maximum best path lengths after all runs for each generation
    

    #overall_avg_best_path_length = np.mean(avg_best_path_length[:num_generations_acoga])
    #overall_max_best_path_length = np.max(max_best_path_length[:num_generations_acoga])
    
    overall_avg_best_path_length = np.mean(avg_best_path_length[:num_runs])
    overall_max_best_path_length = np.max(max_best_path_length[:num_runs])
    
    print("generations:", generations)
    print("overall_avg_best_path_length:", overall_avg_best_path_length)
    print("overall_max_best_path_length:", overall_max_best_path_length)
    
    generations_list.append(generations)
    overall_avg_best_path_lengths_list.append(overall_avg_best_path_length)
    overall_max_best_path_lengths_list.append(overall_max_best_path_length)

# Plotting with magnified figure size
plt.figure(figsize=(12, 8))

# Plotting
plt.plot(generations_list, overall_avg_best_path_lengths_list, label='Avg-Avg Tour Length')
plt.plot(generations_list, overall_max_best_path_lengths_list, label='Max-Avg Tour Length')

# Customize plot appearance
plt.xlabel('Generation')
plt.ylabel('Best Path Length')
plt.title('Overall Best Path Lengths vs. Generation')
plt.title(f'Tour Length vs. Generation\nDataset Instance={file_path}\n\nParameters:n_ants={n_ants},decay={decay}, alpha={alpha}, beta={beta}, ga_population_size={ga_population_size}, ga_elite_size={ga_elite_size}, ga_mutation_rate={ga_mutation_rate}, ga_crossover_prob={ga_crossover_prob}')


plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





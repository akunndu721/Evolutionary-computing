#!/usr/bin/env python
# coding: utf-8

# In[42]:


import random
import matplotlib.pyplot as plt

# Define the Floorplanning constraints
room_data = {
    "Living": {"length": (8, 20), "width": (8, 20), "area": (120, 300), "proportion": 1.5},
    "Kitchen": {"length": (6, 18), "width": (6, 18), "area": (50, 120), "proportion": 1e6},
    "Bath": {"length": (5.5, 5.5), "width": (8.5, 8.5), "area": (None, None), "proportion": None},
    "Hall": {"length": (5.5, 5.5), "width": (3.5, 6), "area": (19, 72), "proportion": 1e6},
    "Bed1": {"length": (10, 17), "width": (10, 17), "area": (100, 180), "proportion": 1.5},
    "Bed2": {"length": (9, 20), "width": (9, 20), "area": (100, 180), "proportion": 1.5},
    "Bed3": {"length": (8, 18), "width": (8, 18), "area": (100, 180), "proportion": 1.5}
}

# Additional constraints
doorway_space = 3.0  # units

# Length and width will be represented by 6 bits each

chromosome_length = 6  

def encode_chromosome():
    chromosome = {}
    for room, constraints in room_data.items():
        min_length, max_length = constraints["length"]
        min_width, max_width = constraints["width"]
        
        length = random.uniform(min_length, max_length)
        width = random.uniform(min_width, max_width)
        
        while True:
            length_bits = format(int(length), f'0{chromosome_length}b')
            width_bits = format(int(width), f'0{chromosome_length}b')
            
            if length_bits != '0' * chromosome_length and width_bits != '0' * chromosome_length:
                break
            else:
                length = random.uniform(min_length, max_length)
                width = random.uniform(min_width, max_width)
        
        chromosome[room] = {"length": length_bits, "width": width_bits}
    return chromosome


def decode_chromosome(chromosome):
    decoded_chromosome = {}
    for room, dimensions in chromosome.items():
        decoded_chromosome[room] = {
            "length": int(dimensions["length"], 2),
            "width": int(dimensions["width"], 2)
        }
    return decoded_chromosome

def fitness_function(chromosome):
    cost = 1
    for room, dimensions in chromosome.items():
        length = int(dimensions["length"], 2)
        width = int(dimensions["width"], 2)
        area = length * width
        
        min_len, max_len, min_width, max_width, min_area, max_area, prop =          room_data[room]["length"][0], room_data[room]["length"][1],         room_data[room]["width"][0], room_data[room]["width"][1],         room_data[room]["area"][0], room_data[room]["area"][1],         room_data[room]["proportion"]  
        
        # Include doorway space constraint
        if room == "Living" or room == "Bed1" or room == "Bed2" or room == "Bed3":
            if (length - doorway_space) < min_len or (length + doorway_space) > max_len                     or (width - doorway_space) < min_width or (width + doorway_space) > max_width                     or min_area and area < min_area or max_area and area > max_area                     or prop and length / width != prop:
                area =  1e6
            cost += 2 * area if room in ('Kitchen', 'Bath') else area
        else:
            if room == "Hall":
                # Check if the space for doorway is available between Bed2, Bed3, and Hall
                bed2_length = int(chromosome["Bed2"]["length"], 2)
                bed2_width = int(chromosome["Bed2"]["width"], 2)
                bed3_length = int(chromosome["Bed3"]["length"], 2)
                bed3_width = int(chromosome["Bed3"]["width"], 2)
                
                if bed2_length + bed3_length >= length + doorway_space and                    bed2_width >= width and bed3_width >= width:
                    pass  # Doorway space is available
                else:
                    area =  1e6  # Adjust area if doorway space is not available
            else:
                if min_len and length < min_len or max_len and length > max_len                         or min_width and width < min_width or max_width and width > max_width                         or min_area and area < min_area or max_area and area > max_area                         or prop and length / width != prop:
                    area =  1e6
                cost += 2 * area if room in ('Kitchen', 'Bath') else area
        
    return cost

def crossover(parent1, parent2, crossover_probability):
    child1 = {}
    child2 = {}
    
    if random.random() < crossover_probability:
        crossover_point = random.choice(list(parent1.keys()))
        for room in room_data.keys():
            if room < crossover_point:
                child1[room] = parent1[room]
                child2[room] = parent2[room]
            else:
                child1[room] = parent2[room]
                child2[room] = parent1[room]
    else:
        child1 = parent1.copy()
        child2 = parent2.copy()
        
    return child1, child2


def mutate(chromosome, mutation_probability):
    mutated_chromosome = chromosome.copy()
    for room, constraints in room_data.items():
        if random.random() < mutation_probability:
            length_bits = ''.join(random.choice('01') for _ in range(chromosome_length))
            width_bits = ''.join(random.choice('01') for _ in range(chromosome_length))
            mutated_chromosome[room]["length"] = length_bits
            mutated_chromosome[room]["width"] = width_bits
    return mutated_chromosome


def genetic_algorithm(population_size, num_generations, crossover_probability, mutation_probability):
    
    # Initialize population
    population = [encode_chromosome() for _ in range(population_size)]

    best_fitness_values = []
    
    # Main loop
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_scores = [fitness_function(chromosome) for chromosome in population]
        best_fitness_values.append(min(fitness_scores))

        # Select parents for crossover
        parents = random.choices(population, weights=fitness_scores, k=population_size)

        # Perform crossover
        new_population = []
        for i in range(0, population_size, 2):
            child1, child2 = crossover(parents[i], parents[i+1], crossover_probability)
            new_population.extend([mutate(child1, mutation_probability), mutate(child2, mutation_probability)])

        # Replace old population with new population
        population = new_population

    # Get the best solution
    best_chromosome = min(population, key=fitness_function)
    best_dimensions = decode_chromosome(best_chromosome)
    return best_chromosome, best_dimensions, best_fitness_values


# Setting GA parameters
    
parameter_combinations = [
    (50, 100, 0.7, 0.001),
    (100, 200, 0.8, 0.002),
    (30, 150, 0.6, 0.001),
    (200, 300, 0.8, 0.002),
    (50, 75, 0.95, 0.05),
    (100, 75, 0.5, 0.01)
]

for params in parameter_combinations:
    population_size, num_generations, crossover_probability, mutation_probability = params
    
    #print(f"Running with parameters: {params}")
    print(f"\033[1;31mRunning with parameters: {params}\033[0m")
    print("\n") 
    best_chromosome, best_floorplan, best_fitness_values = genetic_algorithm(population_size, num_generations, crossover_probability, mutation_probability)

    print(f"\033[1;32mBest Chromosome: {params}\033[0m")
    for room, dimensions in  best_chromosome.items():
        print(f"Room: {room}, Length: {dimensions['length']}, Width: {dimensions['width']}")

    print(f"\033[1;32mBest Floorplan\033[0m")    
    for room, dimensions in  best_floorplan.items():
        length = dimensions['length']
        width = dimensions['width']
        area = length * width
        print(f"Room: {room}, Length: {length}, Width: {width}, Area: {area}")

        #print(f"Room: {room}, Length: {dimensions['length']}, Width: {dimensions['width']}")
    print("\n")  


# In[50]:


import matplotlib.pyplot as plt


# Create a list of labels for the x-axis (parameter combinations)
labels = [f'Params {i+1}' for i in range(len(parameter_combinations))]

# Create a figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot Speed
axs[0].bar(labels, speed_values, color='blue')
axs[0].set_ylabel('Speed (Generations)')
axs[0].set_title('Performance Metrics for Different Parameter Combinations')

# Plot Quality
axs[1].bar(labels, quality_values, color='green')
axs[1].set_ylabel('Quality (Total Area)')

# Plot Reliability
axs[2].bar(labels, reliability_values, color='red')
axs[2].set_ylabel('Reliability')
axs[2].set_xlabel('Parameter Combinations')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[18]:


import matplotlib.pyplot as plt

# Define the dimensions of the rooms
room_dimensions = {
    "Living": (30, 7),
    "Kitchen": (14, 8),
    "Bath": (16, 8),
    "Hall": (4.5, 15),
    "Bed1": (12, 12),
    "Bed2": (13, 12),
    "Bed3": (9.5, 12)
}

# Define the positions of the rooms 
room_positions = {
    "Living": (0, 8),
    "Kitchen": (0, 0),
    "Bath": (14, 0),
    "Hall": (30, 0),
    "Bed1": (0, 15),
    "Bed2": (12, 15),
    "Bed3": (25, 15)
}


fig, ax = plt.subplots()

# Draw rooms as rectangles
for room, dimensions in room_dimensions.items():
    width, height = dimensions
    x, y = room_positions[room]
    ax.add_patch(plt.Rectangle((x, y), width, height, fill=None, edgecolor='b'))

    # Annotate room names
    ax.annotate(room, (x + width/2, y + height/2), color='b', weight='bold', fontsize=8, ha='center', va='center')

    # Annotate dimensions
    #ax.text(x + width/2, y - 1, f'Length: {width}', color='r', weight='bold', fontsize=8, ha='center')
    #ax.text(x - 1, y + height/2, f'Width: {height}', color='r', weight='bold', fontsize=8, va='center')

# Draw corridors (example positions, you'll need to define them)
#corridor_positions = [((5, 5), (5, 15)), ((15, 5), (15, 15)), ((0, 5), (30, 5)), ((0, 15), (30, 15))]
#for (x1, y1), (x2, y2) in corridor_positions:
    #ax.plot([x1, x2], [y1, y2], color='w')

# Set axis limits
ax.set_xlim(0, 34)
ax.set_ylim(0, 27)


# Set aspect of the plot to be equal, so squares look like squares
ax.set_aspect('equal', 'box')

# Show the plot
plt.show()


# In[ ]:





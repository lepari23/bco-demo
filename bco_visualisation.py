import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Reduced number of cities for a cleaner demo
num_cities = 6  # Reduced for simplicity and performance

# Generate random cities (nodes) with some spacing on a 2D plane


def generate_cities(num_cities):
    return np.random.rand(num_cities, 2) * 100 + 50  # Widely spaced points

# Calculate the Euclidean distance between two cities


def calculate_distance(city1, city2):
    return np.linalg.norm(city1 - city2)

# Create a distance matrix for the cities


def create_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = calculate_distance(cities[i], cities[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

# Plot the cities and the edges with weights


def plot_graph(cities, distance_matrix, route=None, highlight_color=None):
    plt.scatter(cities[:, 0], cities[:, 1], c='red')  # Plot cities
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            # Plot the roads between cities
            plt.plot([cities[i, 0], cities[j, 0]], [
                     cities[i, 1], cities[j, 1]], 'gray', linestyle='--')

    # If a route is provided, highlight the path
    if route is not None and highlight_color:
        for i in range(len(route)):
            start_city = cities[route[i]]
            end_city = cities[route[(i + 1) % len(route)]]
            plt.plot([start_city[0], end_city[0]], [start_city[1],
                     end_city[1]], color=highlight_color, linewidth=2)

# Two-opt swap: swap two edges to generate a new route


def two_opt_swap(route):
    new_route = route.copy()
    i, j = sorted(random.sample(range(len(route)), 2))
    # Convert reversed iterator to list
    new_route[i:j] = list(reversed(new_route[i:j]))
    return new_route

# Total route distance based on the distance matrix


def total_route_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))

# BCO function: Bee Colony Optimization for TSP


def bee_colony_optimization(cities, distance_matrix, num_bees, num_iterations, stagnation_limit):
    num_cities = len(cities)
    best_route = np.random.permutation(num_cities)
    best_distance = total_route_distance(best_route, distance_matrix)
    employed_bees = [np.random.permutation(
        num_cities) for _ in range(num_bees)]
    stagnation_count = 0  # To track how long there's been no improvement

    # Bee positions: track their start and target cities along edges
    # Initial positions of bees
    bee_positions = np.array([cities[route[0]] for route in employed_bees])
    # Targets the next city for each bee
    bee_targets = np.array([cities[route[1]] for route in employed_bees])
    # Track progress along the road between cities
    bee_travel_progress = np.zeros(num_bees)

    iteration = 0

    while iteration < num_iterations:
        try:
            # Clear the plot and draw the static graph
            plt.clf()
            plot_graph(cities, distance_matrix)

            # Employed bees explore neighborhood (select new routes)
            improvement_found = False
            for i in range(num_bees):
                new_route = two_opt_swap(employed_bees[i])
                new_distance = total_route_distance(new_route, distance_matrix)
                if new_distance < total_route_distance(employed_bees[i], distance_matrix):
                    employed_bees[i] = new_route
                if new_distance < best_distance:
                    best_route, best_distance = new_route, new_distance
                    improvement_found = True

            # If no improvement found, increment stagnation count
            if not improvement_found:
                stagnation_count += 1
            else:
                stagnation_count = 0  # Reset if there's improvement

            # Stop early if stagnation limit is reached
            if stagnation_count >= stagnation_limit:
                plt.figure()
                plot_graph(cities, distance_matrix, best_route,
                           highlight_color='blue')  # Highlight best route
                plt.title(
                    f"Optimal Solution | Shortest Distance: {best_distance:.2f}\nFound in {iteration} iterations")
                plt.axis('off')  # Turn off axis keys for the final plot
                plt.show()
                return

            # Move bees visibly along the edges
            move_speed = 0.1  # Adjust this to control speed of traversal
            for i, bee_route in enumerate(employed_bees):
                if bee_travel_progress[i] >= 1:
                    current_city_index = bee_route[iteration % num_cities]
                    next_city_index = bee_route[(iteration + 1) % num_cities]
                    bee_positions[i] = cities[current_city_index]
                    bee_targets[i] = cities[next_city_index]
                    bee_travel_progress[i] = 0  # Reset progress

                bee_positions[i] = (1 - bee_travel_progress[i]) * \
                    bee_positions[i] + bee_travel_progress[i] * bee_targets[i]
                # Increment travel progress
                bee_travel_progress[i] += move_speed

                # Plot bee's position (darker yellow and slightly larger)
                # Darker yellow, larger
                plt.scatter(
                    bee_positions[i][0], bee_positions[i][1], c='#FFD700', marker='x', s=120)

            # Display iteration info and best distance so far
            plt.title(
                f"Iteration {iteration} | Shortest Distance: {best_distance:.2f}")
            plt.pause(0.5)  # Pause for a short moment to simulate frame timing

            iteration += 1

        except Exception as e:
            print(f"Error during update: {e}")
            break

    plt.show()


# Initialize the TSP graph
cities = generate_cities(num_cities)
distance_matrix = create_distance_matrix(cities)

# Manually run the BCO optimization
bee_colony_optimization(cities, distance_matrix, num_bees=5,
                        num_iterations=50, stagnation_limit=10)

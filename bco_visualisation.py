import numpy as np
import matplotlib.pyplot as plt
import random
import sys  # To handle command-line arguments

# Handle command-line argument to choose complexity level


def get_complexity():
    if len(sys.argv) < 2 or sys.argv[1] not in ['A', 'B', 'C']:
        print("Usage: python3 bco_visualisation.py [A-C]")
        sys.exit(1)
    return sys.argv[1]

# Set complexity based on command-line argument


def set_complexity(complexity):
    if complexity == 'A':
        # Simple config with weights
        num_cities = 6
        spacing_multiplier = 100
        display_weights = True  # Show weights for configuration A
        print("Running in Simple Mode (A): 6 Cities, With Weights")
    elif complexity == 'B':
        # Moderate config without weights
        num_cities = 10
        spacing_multiplier = 100
        display_weights = False  # No weights for configuration B
        print("Running in Moderate Mode (B): 10 Cities, No Weights")
    else:
        # Complex config without weights
        num_cities = 20  # More cities for complexity
        spacing_multiplier = 75  # Closer spacing for complexity
        display_weights = False  # No weights for configuration C
        print("Running in Complex Mode (C): 20 Cities, No Weights")

    return num_cities, spacing_multiplier, display_weights

# Generate random cities (nodes) with customizable spacing


def generate_cities(num_cities, spacing_multiplier):
    # Widely spaced points
    return np.random.rand(num_cities, 2) * spacing_multiplier + 50

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

# Plot the cities and the edges with optional weights


def plot_graph(cities, distance_matrix, route=None, highlight_color=None, display_weights=False):
    plt.scatter(cities[:, 0], cities[:, 1], c='red')  # Plot cities
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            # Plot the roads between cities
            plt.plot([cities[i, 0], cities[j, 0]], [
                     cities[i, 1], cities[j, 1]], 'gray', linestyle='--')
            # Display the weight (distance) on each road if enabled
            if display_weights:
                mid_x = (cities[i, 0] + cities[j, 0]) / 2
                mid_y = (cities[i, 1] + cities[j, 1]) / 2
                plt.text(
                    mid_x, mid_y, f"{distance_matrix[i, j]:.1f}", fontsize=8, ha='center')

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


def bee_colony_optimization(cities, distance_matrix, num_bees, stagnation_limit, display_weights):
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

    while True:
        try:
            # Clear the plot and draw the static graph
            plt.clf()
            plot_graph(cities, distance_matrix,
                       display_weights=display_weights)

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

            # Stop if stagnation limit is reached (i.e., no improvement for a number of iterations)
            if stagnation_count >= stagnation_limit:
                plt.figure()
                plot_graph(cities, distance_matrix, best_route, highlight_color='blue',
                           display_weights=display_weights)  # Highlight best route
                plt.title(
                    f"Shortest Path: {best_distance:.2f} | {iteration} iterations")
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
                f"Iteration: {iteration} | Shortest Distance: {best_distance:.2f}")
            plt.pause(0.5)  # Pause for a short moment to simulate frame timing

            iteration += 1

        except Exception as e:
            print(f"Error during update: {e}")
            break

    plt.show()


# Main function
if __name__ == '__main__':
    complexity = get_complexity()
    num_cities, spacing_multiplier, display_weights = set_complexity(
        complexity)

    # Initialize the TSP graph
    cities = generate_cities(num_cities, spacing_multiplier)
    distance_matrix = create_distance_matrix(cities)

    # Manually run the BCO optimization
    bee_colony_optimization(cities, distance_matrix, num_bees=5,
                            stagnation_limit=10, display_weights=display_weights)

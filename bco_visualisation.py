import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation

# Turn off interactive mode
plt.ioff()

# Reduced number of cities for a cleaner demo
num_cities = 6  # Reduced for simplicity

# Generate random cities (nodes) with coordinates on a 2D plane


def generate_cities(num_cities):
    return np.random.rand(num_cities, 2) * 100

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
            # Display the weight (distance) on each road
            mid_x = (cities[i, 0] + cities[j, 0]) / 2
            mid_y = (cities[i, 1] + cities[j, 1]) / 2
            plt.text(mid_x, mid_y,
                     f"{distance_matrix[i, j]:.2f}", fontsize=8, ha='center')

    # If a route is provided, highlight the path
    if route is not None and highlight_color:
        for i in range(len(route)):
            start_city = cities[route[i]]
            end_city = cities[route[(i + 1) % len(route)]]
            plt.plot([start_city[0], end_city[0]], [start_city[1],
                     end_city[1]], color=highlight_color, linewidth=2)

# Neighborhood search: Swap two cities to generate a new route


def neighborhood_search(route):
    new_route = route.copy()
    i, j = random.sample(range(len(route)), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# Total route distance based on the distance matrix


def total_route_distance(route, distance_matrix):
    return sum(distance_matrix[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))

# BCO function: Bee Colony Optimization for TSP


def bee_colony_optimization(cities, distance_matrix, num_bees, num_iterations):
    num_cities = len(cities)
    best_route = np.random.permutation(num_cities)
    best_distance = total_route_distance(best_route, distance_matrix)
    employed_bees = [np.random.permutation(
        num_cities) for _ in range(num_bees)]

    # Bee positions: track their start and target cities along edges
    # Initial positions of bees
    bee_positions = np.array([cities[route[0]] for route in employed_bees])
    # Targets the next city for each bee
    bee_targets = np.array([cities[route[1]] for route in employed_bees])

    # For more noticeable movement, track how far along the edge the bee is traveling (between 0 and 1)
    # Track progress along the road between cities
    bee_travel_progress = np.zeros(num_bees)

    def update(frame):
        nonlocal best_route, best_distance, employed_bees, bee_positions, bee_targets, bee_travel_progress

        # Clear the plot and draw the static graph
        plt.clf()
        plot_graph(cities, distance_matrix)

        # Employed bees explore neighborhood (select new routes)
        for i in range(num_bees):
            new_route = neighborhood_search(employed_bees[i])
            new_distance = total_route_distance(new_route, distance_matrix)
            if new_distance < total_route_distance(employed_bees[i], distance_matrix):
                employed_bees[i] = new_route

        # Update best solution
        for bee_route in employed_bees:
            route_distance = total_route_distance(bee_route, distance_matrix)
            if route_distance < best_distance:
                best_route, best_distance = bee_route, route_distance

        # Move bees visibly along the edges
        move_speed = 0.1  # Adjust this to control speed of traversal
        for i, bee_route in enumerate(employed_bees):
            # Check if the bee has reached its target city
            if bee_travel_progress[i] >= 1:
                # Move to the next city in the route
                current_city_index = bee_route[frame % num_cities]
                next_city_index = bee_route[(frame + 1) % num_cities]
                bee_positions[i] = cities[current_city_index]
                bee_targets[i] = cities[next_city_index]
                bee_travel_progress[i] = 0  # Reset progress

            # Move the bee along the edge towards the target city
            bee_positions[i] = (1 - bee_travel_progress[i]) * \
                bee_positions[i] + bee_travel_progress[i] * bee_targets[i]
            bee_travel_progress[i] += move_speed  # Increment travel progress

            plt.scatter(bee_positions[i][0], bee_positions[i][1],
                        c='yellow', marker='x', s=100)  # Plot bee's position

        # Display iteration and best distance
        plt.title(f"Iteration {frame} | Best Distance: {best_distance:.2f}")

        # Once the last frame is reached, plot the optimal solution in another color
        if frame == num_iterations - 1:
            plt.figure()
            plot_graph(cities, distance_matrix, best_route,
                       highlight_color='blue')  # Highlight best route
            plt.title(f"Optimal Solution | Best Distance: {best_distance:.2f}")
            plt.axis('off')  # Turn off axis keys for the final plot

    return update


# Initialize the TSP graph
cities = generate_cities(num_cities)
distance_matrix = create_distance_matrix(cities)

# Create the animation with bees visualized as moving agents
fig, ax = plt.subplots()
plt.axis('off')  # Turn off axis labels to reduce confusion

update_func = bee_colony_optimization(
    cities, distance_matrix, num_bees=5, num_iterations=50)
ani = FuncAnimation(fig, update_func, frames=50, interval=500, repeat=False)

# Ensure that the plot shows
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Generate random cities (nodes) with coordinates on a 2D plane


def generate_cities(num_cities):
    return np.random.rand(num_cities, 2) * 100

# Create a distance matrix for the cities


def create_distance_matrix(cities):
    num_cities = len(cities)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            distance = np.linalg.norm(cities[i] - cities[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix

# Plot the cities and the edges with weights


def plot_graph(cities, distance_matrix):
    plt.scatter(cities[:, 0], cities[:, 1], c='red', label='Cities')
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            plt.plot([cities[i, 0], cities[j, 0]], [
                     cities[i, 1], cities[j, 1]], 'gray', linestyle='--')
            mid_x, mid_y = (cities[i, 0] + cities[j, 0]) / \
                2, (cities[i, 1] + cities[j, 1]) / 2
            plt.text(mid_x, mid_y,
                     f"{distance_matrix[i, j]:.2f}", fontsize=8, ha='center')

    plt.title('TSP Map with Weights (Distances)')
    plt.legend()
    plt.show()

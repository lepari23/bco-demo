# Bee Colony Optimization (BCO) Visualization Demo

This project provides a visual demo of the **Bee Colony Optimization** (BCO) algorithm solving the **Traveling Salesman Problem** (TSP) with varying levels of graph complexity. You can switch between simple, moderate, and complex graphs and observe how bee agents traverse the cities to find the shortest path.

## Features
- **Configurable Complexity**: Choose between three levels of complexity: `A` (simple), `B` (moderate), and `C` (complex).
- **Dynamic Visuals**: Watch as the bee agents explore cities, move along edges, and iteratively optimize their routes.
- **Stopping Condition**: The demo automatically stops once the optimal solution is found (no improvements for multiple iterations).

## Setup Instructions

### 1. Clone the Repository
First, clone the project repository:
```bash
git clone <repository_url>
cd <repository_name>
```

### 2. Set Up a Virtual Environment
For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Widnows:
```cmd
del /s /q *.*
```

### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -r requirements.txt
```
### 4. Run the BCO Visualization

To run the demo, use the following command:
```bash
python3 bco_visualisation.py [A|B|C]
```
Where:

* A runs the visualization in simple mode (fewer cities, visible edge weights).
* B runs the visualization in moderate mode (more cities, no edge weights).
* C runs the visualization in complex mode (maximum number of cities, no edge weights).


## Project Structure

* bco_visualisation.py: The main script for running the BCO visualization.
* requirements.txt: A list of dependencies required for the project.
* README.md: Project instructions and information.

## Troubleshooting

* Ensure your virtual environment is activated before installing dependencies.
* Make sure all dependencies are installed with pip install -r requirements.txt.
* If the visualization doesn’t appear, try upgrading matplotlib:
```
pip install --upgrade matplotlib
```
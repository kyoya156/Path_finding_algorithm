import matplotlib.pyplot as plt
import numpy as np
from grid import Grid
from algorithm import Search_Algorithm

import sys

def visualize_pathfinding(filename: str, algorithm: str, heuristic: str = None):
    """
    Visualize the pathfinding algorithm on a grid.
    """
    
    # Initialize the grid and search algorithm
    grid = Grid(filename)
    search = Search_Algorithm(grid)
    
        # Debug: Print grid information
    print(f"Grid dimensions: {grid.rows} rows x {grid.columns} columns")
    print(f"Start location: {grid.start_location}")
    print(f"End locations: {grid.end_location}")
    print(f"Number of walls: {len(grid.walls)}")

    # Set algorithm and heuristic
    search.algorithm = algorithm
    if heuristic:
        if heuristic not in ["manhattan", "euclidean", "chebyshev"]:
            raise ValueError(f"Invalid heuristic: {heuristic}")
        search.set_heuristic(heuristic)
    
    # Run the search
    result, end_reached = search.search()
    
    # Get the path 
    path_coords = search.get_path()

    # Calculate figure size based on grid dimensions
    # Make each cell roughly square
    cell_size = 0.4  # Size of each cell in inches
    fig_width = max(8, grid.columns * cell_size)
    fig_height = max(6, grid.rows * cell_size)

    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a color map for the grid
    # 0 = free space (white), 1 = wall (black), 2 = visited (light blue), 
    # 3 = path (green), 4 = start (red), 5 = end (blue)
    grid_visual = np.zeros((grid.rows, grid.columns))

     # Helper function to check if coordinates are valid
    def is_valid_coordinate(x, y):
        return 0 <= x < grid.columns and 0 <= y < grid.rows
    
    # Mark walls
    walls_out_of_bounds = 0
    for wall in grid.walls:
        x, y = wall
        if is_valid_coordinate(x, y):
            grid_visual[y, x] = 1  # Note: numpy uses [row, col] = [y, x]
        else:
            walls_out_of_bounds += 1
            print(f"Warning: Wall at ({x}, {y}) is out of bounds!")
    
    if walls_out_of_bounds > 0:
        print(f"Total walls out of bounds: {walls_out_of_bounds}")

    # Mark visited nodes (excluding path)
    visited_nodes = search.get_visited_nodes()
    for node in visited_nodes:
        x, y = node
        if 0 <= y < grid.rows and 0 <= x < grid.columns:
            if node not in path_coords:  # Don't overwrite path
                grid_visual[y, x] = 2
        else:
            print(f"Warning: Visited node at ({x}, {y}) is out of bounds!")
    # Mark the path
    for path_coord in path_coords:
        x, y = path_coord
        if 0 <= y < grid.rows and 0 <= x < grid.columns:
            grid_visual[y, x] = 3
        else:
            print(f"Warning: Path coordinate at ({x}, {y}) is out of bounds!")
    
    # Mark start location
    start_x, start_y = grid.start_location
    if is_valid_coordinate(start_x, start_y):
        grid_visual[start_y, start_x] = 4
    else:
        print(f"Warning: Start location ({start_x}, {start_y}) is out of bounds!")
    
    # Mark end locations
    for end_loc in grid.end_location:
        end_x, end_y = end_loc
        if is_valid_coordinate(end_x, end_y):
            grid_visual[end_y, end_x] = 5
        else:
            print(f"Warning: End location ({end_x}, {end_y}) is out of bounds!")
    
    # Create color map
    colors = ['white', 'black', 'lightblue', 'lightgreen', 'red', 'blue']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    # Display the grid
    im = ax.imshow(grid_visual, cmap=cmap, vmin=0, vmax=5, aspect='equal')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Set major ticks for better readability
    ax.set_xticks(np.arange(0, grid.columns, max(1, grid.columns//10)))
    ax.set_yticks(np.arange(0, grid.rows, max(1, grid.rows//10)))

    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    title = f'Pathfinding Visualization\nAlgorithm: {algorithm}'
    if heuristic:
        title += f', Heuristic: {heuristic}'
    ax.set_title(title)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Free Space'),
        plt.Rectangle((0,0),1,1, facecolor='black', label='Wall'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Visited'),
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Path'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='Start'),
        plt.Rectangle((0,0),1,1, facecolor='blue', label='End')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add statistics
    stats_text = f"Grid Size: {grid.rows}×{grid.columns}\n"
    stats_text += f"Nodes Explored: {search.nodes_count}\n"
    stats_text += f"Path Length: {len(path_coords) - 1 if path_coords else 0}\n"
    stats_text += f"Path Found: {'Yes' if end_reached else 'No'}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def main():
    """Main function for visualization."""
    if len(sys.argv) < 3:
        print("Usage: python visualize_main.py <filename> <algorithm> [heuristic]")
        print("Example: python visualize_main.py grid.txt AS manhattan")
        print("Available algorithms: DFS, BFS, AS, GBFS, CUS1, CUS2")
        print("Heuristic options: manhattan, euclidean, chebyshev")
        sys.exit(1)
    
    filename = sys.argv[1]
    algorithm = sys.argv[2]
    heuristic = sys.argv[3] if len(sys.argv) > 3 else None
    
    try:
        visualize_pathfinding(filename, algorithm, heuristic)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
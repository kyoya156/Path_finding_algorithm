from grid import Grid
from algorithm import Search_Algorithm

def solve_problem(filename: str, algorithm: str, heuristic: str = None):
    # Initialize the grid from the file
    grid = Grid(filename)
    
    # Initialize the search algorithm with the grid
    search = Search_Algorithm(grid)

    # Set the algorithm to be used
    search.algorithm = algorithm

    # Set the heuristic for the algorithm
    if heuristic:
        if heuristic not in ["manhattan", "euclidean", "chebyshev"]:
            raise ValueError(f"Invalid heuristic: {heuristic}. Available options are: manhattan, euclidean, chebyshev.")
        
        search.set_heuristic(heuristic)

    result, end_reached = search.search()

    # Handle the case where search returns different formats
    if end_reached is None:
        # If no path is found return this
        print(filename + " " + algorithm)
        print("No path found.", search.nodes_count)
        return
    
    print(filename + " " + algorithm)
    if not result:
        print("No path found.", search.nodes_count)
        return
    print(f"<Node {end_reached}> {search.nodes_count}")
    print(result)

def main():
    """Main function to run the pathfinding algorithm."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python <main_script>.py <filename> <algorithm> [heuristic]")
        print("Example: python main.py grid.txt dfs")
        print("Available algorithms: DFS, BFS, AS, GBFS, CUS1, CUS2")
        print("Heuristic options: manhattan, euclidean, chebyshev(heuristics is optional you can ignore) ")
        sys.exit(1)

    filename = sys.argv[1]
    algorithm = sys.argv[2]
    heuristic = sys.argv[3] if len(sys.argv) > 3 else None


    try:
        solve_problem(filename, algorithm, heuristic)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
# This code is the main entry point for the pathfinding application.
import math
from collections import deque

class Search_Algorithm:
    """Search Algorithm class for pathfinding"""

    def __init__(self, grid):
        self.grid = grid
        self.valid_locations = []
        self.path = []
        self.visited = set()  # Set to keep track of visited nodes
        self.heuristic = "manhattan"  # Default heuristic
        self.algorithm = "DFS"  # Default algorithm
        self.nodes_count = 0  # Counter for nodes created during search

    def get_visited_nodes(self) -> set:
        # Return the set of visited nodes
        return self.visited
    
    def get_path(self) -> list:
        # Return the path found by the search algorithm
        return self.path
    
    def get_valid_locations(self) -> list:
        # Return the list of valid locations
        return self.valid_locations

    def is_valid_location(self, x: int, y: int) -> bool:
        # Check if the location is either within the grid or is a wall
        if 0 <= x < self.grid.columns and 0 <= y < self.grid.rows and (x, y) not in self.grid.walls:
              return True
        return False
          
    def is_end_location(self, x: int, y: int) -> bool:
        # Check if the location is one of the end locations
        return (x, y) in self.grid.end_location

    def get_neighbors_grids(self, x: int, y: int) -> list:
        # Get the valid neighbors of a cell
        neighbors = []
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:  # Right, Down, Left, Up
            if self.is_valid_location(x + dx, y + dy):
                neighbors.append((x + dx, y + dy))
        return neighbors

    def path_to_directions(self, path: list) -> list:
        # Convert a path of coordinates to a list of directions
        if len(path) < 2:
            return []
        
        
        directions_map = {
            (1, 0): "Right",
            (0, 1): "Down",
            (-1, 0): "Left",
            (0, -1): "Up"
        }

        directions = []
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            direction = directions_map.get((dx, dy))
            if direction:
                directions.append(direction)
        return directions
    
    '''
    Heuristic functions
    '''

    def manhattan_distance(self, position: tuple, end: tuple) -> int:
        # Calculate the Manhattan distance between two points
        return abs(position[0] - end[0]) + abs(position[1] - end[1])

    def euclidean_distance(self, position: tuple, end: tuple) -> float:
        # Calculate the Euclidean distance between two points
        return math.sqrt((position[0] - end[0]) ** 2 + (position[1] - end[1]) ** 2)

    def chebyshev_distance(self, position: tuple, end: tuple) -> int:
        # Calculate the Chebyshev distance between two points
        return max(abs(position[0] - end[0]), abs(position[1] - end[1]))
    
    def set_heuristic(self, heuristic: str):
        # Set the heuristic function to be used
        if heuristic in ["manhattan", "euclidean", "chebyshev"]:
            self.heuristic = heuristic
        else:
            raise ValueError("Invalid heuristic. Choose from 'manhattan', 'euclidean', or 'chebyshev'.")
        
    def get_heuristic(self, position: tuple, end_locations: list) -> float:
        # Get the min heuristic value from the current position to any of the end locations
        # optimized to handle both single and multiple end locations
        # which counter to the bug found by extensive testing
        if isinstance(end_locations, list):
            min_heuristic = float('inf')
            for end in end_locations:
                if self.heuristic == "manhattan":
                    heuristic_value = self.manhattan_distance(position, end)
                elif self.heuristic == "euclidean":
                    heuristic_value = self.euclidean_distance(position, end)
                elif self.heuristic == "chebyshev":
                    heuristic_value = self.chebyshev_distance(position, end)
                else:
                    raise ValueError("Heuristic not set or invalid.")
                min_heuristic = min(min_heuristic, heuristic_value)
            return min_heuristic
        elif isinstance(end_locations, tuple):
            # Single end location (single tuple)
            if self.heuristic == "manhattan":
                return self.manhattan_distance(position, end_locations)
            elif self.heuristic == "euclidean":
                return self.euclidean_distance(position, end_locations)
            elif self.heuristic == "chebyshev":
                return self.chebyshev_distance(position, end_locations)
            else:
                raise ValueError("Heuristic not set or invalid.")
        else:
            raise ValueError("End locations must be a list of tuples or a single tuple.")
    '''Pathfinding methods'''

    '''Blind search methods'''

    def dfs(self, position: tuple, end: tuple) -> list:
        # Depth-first search algorithm
        self.visited = set()  
        self.visited.add(position)
        stack = [(position, [position])]
        self.nodes_count = 1  # Initialize the node count
        # Initialize the node count to 1 for the starting position

        while stack:
            (current, path) = stack.pop()

            if current in end:
                self.path = path  # Store the path
                return self.path_to_directions(path), current
            
            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.nodes_count += 1  # Increment the node count
                    stack.append((neighbor, path + [neighbor]))
        self.path = path  # Store the last path if no end is found
        return [], None  # Return empty path if no end is found

    def bfs(self, position: tuple, end: tuple) -> list:
        # Breadth-first search algorithm
        self.visited = set()  
        self.visited.add(position)
        queue = deque([(position, [position])])
        self.nodes_count = 1  # Initialize the node count

        while queue:
            (current, path) = queue.popleft()
            if current in end:
                self.path = path
                return self.path_to_directions(path), current
            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.nodes_count += 1  # Increment the node count
                    queue.append((neighbor, path + [neighbor]))
        self.path = path  # Store the last path if no end is found
        return [], None

    '''Informed search methods'''

    def gbfs(self, position: tuple, end: tuple) -> list:
        # Greedy best-first search algorithm
        from queue import PriorityQueue
        
        #there might be cases where end = start, so we need to handle that
        if position in end:
            self.nodes_count = 1  # Only the starting position is visited
            return self.path_to_directions([position]), position

        self.visited = set()  # Reset visited nodes
        self.visited.add(position)
        self.nodes_count = 1  # Initialize the node count
        pq = PriorityQueue()
        pq.put((0, 0, (position, [position])))
        counter = 1  # Counter to break ties for equal priorities

        while not pq.empty():
            (_, _, (current, path)) = pq.get()

            if current in end:
                self.path = path
                return self.path_to_directions(path), current

            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in self.visited:
                    self.nodes_count += 1  # Increment the node count
                    self.visited.add(neighbor)
                    heuristic_cost = self.get_heuristic(neighbor, end)
                    pq.put((heuristic_cost, counter, (neighbor, path + [neighbor])))
                    counter += 1
        self.path = path  # Store the last path if no end is found
        return [], None  # Return empty path if no end is found

    def a_star(self, position: tuple, end: tuple) -> list:
        # A* search algorithm
        from queue import PriorityQueue

        # Handle case where start equals end
        if position in end:
            self.nodes_count = 1  # Only the starting position is visited
            self.visited.add(position)
            return self.path_to_directions([position]), position
    
        pq = PriorityQueue()
        pq.put((0, 0, (position, [position], 0)))
        counter = 1  # Counter to break ties for equal priorities
        self.nodes_count = 1  # Initialize the node count
        self.visited = set()

        while not pq.empty():
            (f_cost, _, (current, path, g_cost)) = pq.get()

            if current in self.visited:
                continue
            self.visited.add(current)

            if current in end:
                self.path = path
                return self.path_to_directions(path), current

            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in self.visited:
                    self.nodes_count += 1  # Increment the node count
                    heuristic_cost = self.get_heuristic(neighbor, end)
                    new_f_cost = g_cost + 1 + heuristic_cost
                    pq.put((new_f_cost, counter, (neighbor, path + [neighbor], g_cost + 1)))
                    counter += 1
        self.path = path
        return [], None
    
    '''Custom search methods'''
    #custom search 1 is an uninformed search algorithm
    def custom_search_1(self, position: tuple, end: tuple) -> list:
        # Dijkstra's algorithm for finding the shortest path in a grid based world
        # This algorithm does not use heuristics and finds the shortest path based on cost
        from queue import PriorityQueue

        if position in end:
            self.nodes_count = 1
            self.visited = set()
            self.visited.add(position)
            return self.path_to_directions([position]), position
        
        self.visited = set()  # Reset visited nodes
        pq = PriorityQueue()
        pq.put((0, (position, [position])))
        self.nodes_count = 1  # Initialize the node count

        while not pq.empty():
            (cost, (current, path)) = pq.get()
            
            if current in self.visited:
                continue

            self.visited.add(current)

            if current in end:
                self.path = path
                return self.path_to_directions(path), current
            
            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in self.visited:
                    self.nodes_count += 1  # Increment the node count
                    pq.put((cost + 1, (neighbor, path + [neighbor])))

        self.path = path  # Store the last path if no end is found
        return [], None  # Return empty path if no end is found

    #custom search 2 is an informed search algorithm
    def custom_search_2(self, position: tuple, end: tuple) -> list:
        # Iterative Deepening A* search algorithm
        # This algorithm combines the iterative deepening depth-first search with a heuristic to find the path

        def idas_cost_calculation(current, path, g_cost, threshold, visited_in_path):
            f_cost = g_cost + self.get_heuristic(current, end)

            #for tracking the visited nodes
            if current not in self.visited:
                self.visited.add(current)


            if f_cost > threshold:
                return f_cost, None
            
            if current in end:
                return f_cost, path
            
            min_threshold = float('inf')
            for neighbor in self.get_neighbors_grids(*current):
                if neighbor not in visited_in_path:
                    self.nodes_count += 1
                    new_visited = visited_in_path | {neighbor}
                    result, found_path = idas_cost_calculation(neighbor, path + [neighbor], g_cost + 1, threshold, new_visited)
                    
                    if found_path:
                        return result, found_path
                    
                    if result < min_threshold:
                        min_threshold = result

            return min_threshold, None
        
        #Handle case where start equals end
        if position in end:
            self.nodes_count = 1
            return self.path_to_directions([position]), position
        
        # Start the IDA* search
        self.nodes_count = 1
        threshold = self.get_heuristic(position, end)

        while True:
            result, found_path = idas_cost_calculation(position, [position], 0, threshold,{position})

            if found_path:
                self.path = found_path
                return self.path_to_directions(found_path), found_path[-1]
            
            if result == float('inf'):
                # If the result is infinity, it means no path was found
                self.path = []
                return [], None
            threshold = result

    '''Search method to call the appropriate search algorithm'''
    def search(self) -> list:
        # General search method to call the appropriate search algorithm
        position = self.grid.start_location

        if not self.grid.end_location:
            return []  # No end location specified

        '''Old bugged code for applying heuristics to the end location
        which was found by extensive testing and quickly fixed and i will leave it here for reference
        to show just that(this code combined with the old get_heuristic function caused a bug 
        where  because of heuristic applied in search function to allow informed search to narrow down the 
        end location to 1 loc if there are more than 1 that causes the informed algor to run aimlessly and 
        not reach another end location if that narrowed down location is unreachable but the other ends are reachable.'''
        # apply heuristics to the end location
        # if there are multiple end locations, choose the closest one
        # if len(self.grid.end_location) == 1:
        #     end = self.grid.end_location[0]
        # else:
        #     # Find the closest end location based on the heuristic
        #     min_distance = float('inf')
        #     closest_end = None
        #     for end_location in self.grid.end_location:
        #         distance = self.get_heuristic(position, end_location)
        #         if distance < min_distance:
        #             min_distance = distance
        #             closest_end = end_location
        #     end = closest_end
        # blind search algorithms goes before applying heuristics

        if self.algorithm == "DFS"  or self.algorithm == "dfs":
            return self.dfs(position, self.grid.end_location)
        elif self.algorithm == "BFS" or self.algorithm == "bfs":
            return self.bfs(position, self.grid.end_location)
        elif self.algorithm == "CUS1" or self.algorithm == "cus1":
            return self.custom_search_1(position, self.grid.end_location)
        elif self.algorithm == "GBFS" or self.algorithm == "gbfs":
            return self.gbfs(position, self.grid.end_location)
        elif self.algorithm == "AS" or self.algorithm == "as":
            return self.a_star(position, self.grid.end_location)
        elif self.algorithm == "CUS2" or self.algorithm == "cus2":
            return self.custom_search_2(position, self.grid.end_location)
        else:
            raise ValueError("Invalid search algorithm specified.")

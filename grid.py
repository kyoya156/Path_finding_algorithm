# grid.py
# a grid in a 2D space with specified start and end locations, walls, rows, columns, and cell size.
class Grid:
    """ initialize grid world from a text file """

    def __init__(self, filename: str):
        self.start_location = ()  # Start location as a tuple (x, y)
        self.end_location = []  # List of end locations as tuples [(x1, y1), (x2, y2), ...]
        self.walls = set()  # Set of (x, y) coordinates that are blocked
        self.rows = 0
        self.columns = 0

        self.read_text_file(filename)

    def read_text_file(self, filename: str):
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        
            """According to the requirements the 
            first line declares the size of the grid world
            second line declares the start location
            third line declares the end location
            the rest of the lines declare the walls"""

            # first grid size
            line_one = lines[0].strip('[]').split(',')
            self.rows = int(line_one[0])
            self.columns = int(line_one[1])

            # second start location
            line_two = lines[1].strip('()').split(',')
            x, y = (int(line_two[0]), int(line_two[1]))
            self.start_location = (x, y)

            # third end location
            line_three = lines[2]
            ends = line_three.split('|')
            for end in ends:
                end = end.strip()
                x, y = map(int, end.strip('()').split(','))
                self.end_location.append((x, y))

            # the rest of the lines declare the walls
            for line in range(3, len(lines)):
                if lines[line].strip():  # Check if the line is not empty
                    wall_data = lines[line].strip('()').split(',')
                    x, y, w, h = int(wall_data[0]), int(wall_data[1]), int(wall_data[2]), int(wall_data[3])

                    # Add walls to the set
                    for i in range(x, x + w):
                        for j in range(y, y + h):
                            self.walls.add((i, j))
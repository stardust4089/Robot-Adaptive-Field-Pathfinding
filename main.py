import json
import math
from queue import PriorityQueue

class Shape:
    # POINT INPUT MUST BE IN COUNTER CLOCKWISE ORDER
    def __init__(self, p1, p2, p3, p4):
        self.points = [p1, p2, p3, p4]
 

# Define your grid dimensions, start point, and goal point
# decimeters    
GRID_WIDTH = (165)
GRID_HEIGHT = (80)
START = (0, 0)
GOAL = (160, 70)

# Define your obstacle positions
# You can represent obstacles as a list of coordinates or any other suitable data structure
NO_GO = []
OBSTACLES = []
# Define movement costs (you can adjust these as needed)
MOVE_STRAIGHT_COST = 1
MOVE_DIAGONAL_COST = math.sqrt(2)

# Define the output file path
OUTPUT_FILE = "path_output.path"

# Define a class to represent each node on the grid
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')  # cost from start node
        self.h = float('inf')  # estimated cost to goal node
        self.f = float('inf')  # total cost
        self.parent = None
        self.obstacle = False  # whether the node is an obstacle

    def __lt__(self, other):
        return self.f < other.f



# Create a grid of nodes
grid = [[Node(x, y) for y in range(GRID_HEIGHT)] for x in range(GRID_WIDTH)]

# Set the obstacle flag for nodes that contain obstacles
for obstacle in OBSTACLES:
    grid[obstacle[0]][obstacle[1]].obstacle = True

# Heuristic function for estimating the distance between two nodes (Manhattan distance)
def heuristic(node_a, node_b):
    return abs(node_a.x - node_b.x) + abs(node_a.y - node_b.y)
def changed_direction(curr_node, prev_node, prev_prev_node):
    x_delta = curr_node.x - prev_node.x
    y_delta = curr_node.y - prev_node.y
    old_x_delta = prev_node.x - prev_prev_node.x
    old_y_delta = prev_node.y - prev_prev_node.y
    if (x_delta == old_x_delta and y_delta == old_y_delta):
        return False
    return True
#im gay
def reverse_linked_list(head):
    prev_node = None
    current_node = head

    while current_node is not None:
        next_node = current_node.parent
        current_node.parent = prev_node
        prev_node = current_node
        current_node = next_node

    # Return the new head of the reversed linked list
    return prev_node

# Function to generate the path from the goal node to the start node
def generate_path(end_node):
    path = []
    current_node = reverse_linked_list(end_node)
    prev_control = None
    
    prev_node = None
    prev_prev_node = None
    delay = 2
    did_change_dir = True

    while current_node is not None:
        delay -= 1
        
        anchor_point = {
            "x": current_node.x / 10,
            "y": current_node.y / 10
        }
        
        waypoint = {
            "anchorPoint": anchor_point,
            "prevControl": prev_control,
            "nextControl": None,
            "holonomicAngle": 0,
            "isReversal": False,
            "velOverride": None,
            "isLocked": False,
            "isStopPoint": False,
            "stopEvent": {
                "names": [],
                "executionBehavior": "parallel",
                "waitBehavior": "none",
                "waitTime": 0
            }
        }
        if (did_change_dir or current_node.parent is None):
            path.append(waypoint)
        prev_control = anchor_point
        
        prev_prev_node = prev_node
        prev_node = current_node
        current_node = current_node.parent


        if (current_node is not None and prev_node is not None and prev_prev_node is not None and (delay <= 0)):
            did_change_dir = changed_direction(current_node, prev_node, prev_prev_node)
    # Update the nextControl values
    for i in range(len(path)-1):
        path[i]["nextControl"] = path[i+1]["anchorPoint"]
    
    # Set prevControl to None for the first waypoint
    path[0]["prevControl"] = None
    
    # Set nextControl to None for the last waypoint
    path[-1]["nextControl"] = None
    
    return path

# A* algorithm implementation
def astar():
    # Create the priority queue
    open_set = PriorityQueue()
    
    # Initialize the start node
    start_node = grid[START[0]][START[1]]
    start_node.g = 0
    start_node.h = heuristic(start_node, grid[GOAL[0]][GOAL[1]])
    start_node.f = start_node.g + start_node.h
    open_set.put(start_node)
    
    while not open_set.empty():
        # Get the node with the lowest f-score from the open set
        current_node = open_set.get()
        
        # Check if the goal has been reached
        if current_node.x == GOAL[0] and current_node.y == GOAL[1]:
            return current_node
        
        # Explore the neighboring nodes
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                new_x = current_node.x + dx
                new_y = current_node.y + dy
                
                if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
                    neighbors.append(grid[new_x][new_y])
        
        for neighbor in neighbors:
            if neighbor.obstacle:
                continue
            
            # Calculate the tentative g-score for the neighbor
            if abs(neighbor.x - current_node.x) + abs(neighbor.y - current_node.y) == 1:
                tentative_g_score = current_node.g + MOVE_STRAIGHT_COST
            else:
                tentative_g_score = current_node.g + MOVE_DIAGONAL_COST
            
            if tentative_g_score < neighbor.g:
                # Update the neighbor's properties
                neighbor.parent = current_node
                neighbor.g = tentative_g_score
                neighbor.h = heuristic(neighbor, grid[GOAL[0]][GOAL[1]])
                neighbor.f = neighbor.g + neighbor.h
                
                # Add the neighbor to the priority queue
                if neighbor not in open_set.queue:
                    open_set.put(neighbor)
    
    # No path found
    return None

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


for spe in NO_GO:
    for i in range(4):
        p1 = spe.points[i]
        p2 = spe.points[(i+1)%4]
        OBSTACLES += bresenham_line(p1[0], p1[1], p2[1], p2[1])


# Run the A* algorithm
path = astar()

# Generate the output file in .path format
output_data = {
    "waypoints": [],
    "markers": []
}

if path is not None:
    output_data["waypoints"] = generate_path(path)
else:
    print("No path found")

# Write the output file
with open(OUTPUT_FILE, 'w') as file:
    json.dump(output_data, file, indent=2)


def main():
    astar()
    


if __name__ == "__main__":
    main()

import json
import math
from flask import Flask, render_template, request, jsonify, send_file, make_response
import os
import json
import math
from queue import PriorityQueue

class Shape:
    # POINT INPUT MUST BE IN COUNTER CLOCKWISE ORDER
    def __init__(self, p1, p2, p3, p4):
        self.points = [p1, p2, p3, p4]

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


app = Flask(__name__)

# Create an empty array to store the clicked points
obstacles = [(164,0)]

# Define the size of the grid
grid_width = 165
grid_height = 80
square_size = 0  # Size of each square in pixels


ppfile = "a"
# Define your grid dimensions, start point, and goal point
# decimeters    
START = (0, 0)
GOAL = (160, 70)
# Define movement costs (you can adjust these as needed)
MOVE_STRAIGHT_COST = 1
MOVE_DIAGONAL_COST = math.sqrt(2)

# Define the output file path
OUTPUT_FILE = "path_output.path"


# Define the colors
color_normal = "gray"  # Color for normal squares
color_clicked = "blue"  # Color for clicked squares
color_obstacle = "red"  # Color for obstacle squares

# Variables to track the shape construction
shape_points = []  # Stores the points for shape construction

# Function to handle mouse clicks
@app.route('/')
def index():
    return render_template('index.html', obstacles=obstacles, ppfile=ppfile)

@app.route('/add_point', methods=['POST'])
def add_point():
    global obstacles, shape_points

    x = int(request.form['x'])
    y = int(request.form['y'])
    y = y % 80
    point = (x, y)
    
    if point in obstacles:
        obstacles.remove(point)
    else:
        obstacles.append(point)

    return json.dumps(obstacles)

@app.route('/get_obstacles')
def get_obstacles():
    obs = obstacles
    return jsonify(obstacles=obs)

# Endpoint to clear the obstacles array
@app.route('/clear_obstacles', methods=['POST'])
def clear_obstacles():
    global obstacles
    obstacles = []  # Clear the obstacles array
    return jsonify(message='Obstacles cleared successfully')

@app.route('/construct_shape', methods=['POST'])
def construct_shape():
    global shape_points, obstacles
    temp = []
    temp = json.loads(request.form['shape_points'])
    #shape_points = json.loads(request.form['points'])

    # Get the coordinates of the shape points
    x1, y1 = temp[0]
    x2, y2 = temp[1]
    x3, y3 = temp[2]
    x4, y4 = temp[3]

    points = []

    points += bresenham_line(x1, y1, x2, y2)
    points += bresenham_line(x2, y2, x3, y3)
    points += bresenham_line(x3, y3, x4, y4)
    points += bresenham_line(x4, y4, x1, y1)
    points += bresenham_line(x1, y1, x3, y3)
    points += bresenham_line(x2, y2, x4, y4)

    for point in points:
        x, y = point
        obstacles.append(point)

    obstacles = list(set(obstacles))
    print("Obstacles:", obstacles)
    shape_points = []
    return json.dumps(obstacles)

def construct_shape(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    points = []

    points += bresenham_line(x1, y1, x2, y2)
    points += bresenham_line(x2, y2, x3, y3)
    points += bresenham_line(x3, y3, x4, y4)
    points += bresenham_line(x4, y4, x1, y1)
    points += bresenham_line(x1, y1, x3, y3)
    points += bresenham_line(x2, y2, x4, y4)

    for point in points:
        x, y = point
        obstacles.append(point)

    obstacles = list(set(obstacles))
    print("Obstacles:", obstacles)
    return json.dumps(obstacles)

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

        points.append((x0 + 1, y0))
        points.append((x0, y0 + 1))

    return points

def config():
    global grid
    # Create a grid of nodes
    grid = [[Node(x, y) for y in range(grid_height)] for x in range(grid_width)]

    # Set the obstacle flag for nodes that contain obstacles
    for obstacle in obstacles:
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
                
                if 0 <= new_x < grid_width and 0 <= new_y < grid_height:
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

@app.route('/runny_nose', methods=['POST', 'GET'])
def run():
    config()

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
    with open("path_output.path", 'w') as file:
        json.dump(output_data, file, indent=2)
    
    file_path = 'path_output.path'

    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    # Calculate the square size based on the screen size and grid size
    square_size = min(math.floor(800 / grid_height), math.floor(1650 / grid_width))

    app.run()

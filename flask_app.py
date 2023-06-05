import json
import math
from flask import Flask, render_template, request, jsonify, send_file
from flask import Response
import cv2
import json
import math
from queue import PriorityQueue
import pandas as pd
import TFLite_detection_webcam
import math


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.g = float('inf')  # cost from start node``
        self.h = float('inf')  # estimated cost to goal node
        self.f = float('inf')  # total cost
        self.parent = None
        self.obstacle = False

    def __lt__(self, other):
        return self.f < other.f


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

# Create an empty array to store the clicked points
obstacles = [[-1, -1]]
# handles robots
moving_obstacles = [[1, 1]]

# Define the size of the grid (decimeeters)
grid_width = 166
grid_height = 81
square_size = 0  # Size of each square in pixels

new_points = []
grid = [[Node(x, y) for y in range(grid_height)] for x in range(grid_width)]

camera = cv2.VideoCapture("D:/xyzab.mp4")  # use 0 for web camera
moving_obstacles = []

# decimeters
START = (6, 1)
GOAL = (164, 80)

MOVE_STRAIGHT_COST = 1
MOVE_DIAGONAL_COST = math.sqrt(2)

OUTPUT_FILE = "path_output.path"


@app.route('/')
def index():
    frame = None  # this is a load-bearing line of code
    return render_template('index.html', obstacles=obstacles, moving_obstacles=moving_obstacles, new_points=new_points, frame=frame)


@app.route('/upload_json', methods=['POST'])
def upload_json():
    global obstacles
    if 'jsonFile' in request.files:
        json_file = request.files['jsonFile']
        try:
            json_data = json.load(json_file)

            obstacles.extend(json_data)
            print(obstacles)
            return jsonify({'message': 'JSON file uploaded successfully.'}), 200
        except json.JSONDecodeError:
            return jsonify({'message': 'Invalid JSON file.'}), 400
    else:
        return jsonify({'message': 'No JSON file uploaded.'}), 400


@app.route('/add_point', methods=['POST'])
def add_point():
    x = int(request.form['x'])
    clamp(x, 0, grid_width - 1)
    y = int(request.form['y'])
    clamp(y, 0, grid_height - 1)
    point = [x, y]
    if [-1, -1] in obstacles:
        obstacles.remove([-1, -1])
    if point in obstacles:
        obstacles.remove(point)
        print("Removed point:", point)
        return json.dumps((-1, -1))
    else:
        obstacles.append(point)
        print("Added point:", point)
        return json.dumps(point)


@app.route('/get_obstacles')
def get_obstacles():
    obs = obstacles
    return jsonify(obstacles=obs)


@app.route('/get_moving_obstacles')
def get__moving_obstacles():
    movobs = moving_obstacles
    return jsonify(moving_obstacles=movobs)


@app.route('/get_fps')
def get_fps():
    fps = TFLite_detection_webcam.get_fps()
    return jsonify(fps=fps)


@app.route('/get_new_obstacles')
def get_new_obstacles():
    new_points = new_points
    return jsonify(new_points=new_points)


@app.route('/clear_obstacles', methods=['POST'])
def clear_obstacles():
    global obstacles
    obstacles = []
    return jsonify(message='Obstacles cleared successfully')


@app.route('/construct_shape', methods=['POST', 'GET'])
def construct_shape():
    global shape_points, obstacles
    if [-1, -1] in obstacles:
        obstacles.remove([-1, -1])
    temp = []
    temp = json.loads(request.form['shape_points'])

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
        x = clamp(x, 0, grid_width - 1)
        y = clamp(y, 0, grid_height - 1)
        obstacles.append((x, y))
    obstacles = pd.Series(obstacles).drop_duplicates().tolist()
    print("New Obstacles:", points)
    new_points = points
    return json.dumps(new_points)


def construct_shape(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    points = []
    temp = []
    points += bresenham_line(x1, y1, x2, y2)
    points += bresenham_line(x2, y2, x3, y3)
    points += bresenham_line(x3, y3, x4, y4)
    points += bresenham_line(x4, y4, x1, y1)
    points += bresenham_line(x1, y1, x3, y3)
    points += bresenham_line(x2, y2, x4, y4)

    for point in points:
        x, y = point
        x = clamp(x, 0, grid_width - 1)
        y = clamp(y, 0, grid_height - 1)
        temp.append((x, y))

    temp = pd.Series(temp).drop_duplicates().tolist()

    return temp


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
    for obstacle in obstacles:
        print(obstacle[0], " , ", obstacle[1])
        grid[obstacle[0]][obstacle[1]].obstacle = True


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

# im gay
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
            did_change_dir = changed_direction(
                current_node, prev_node, prev_prev_node)
    for i in range(len(path)-1):
        path[i]["nextControl"] = path[i+1]["anchorPoint"]

    path[0]["prevControl"] = None

    path[-1]["nextControl"] = None

    return path


def astar():
    open_set = PriorityQueue()

    start_node = grid[START[0]][START[1]]
    start_node.g = 0
    start_node.h = heuristic(start_node, grid[GOAL[0]][GOAL[1]])
    start_node.f = start_node.g + start_node.h
    open_set.put(start_node)

    while not open_set.empty():
        current_node = open_set.get()

        if current_node.x == GOAL[0] and current_node.y == GOAL[1]:
            return current_node

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

            if abs(neighbor.x - current_node.x) + abs(neighbor.y - current_node.y) == 1:
                tentative_g_score = current_node.g + MOVE_STRAIGHT_COST
            else:
                tentative_g_score = current_node.g + MOVE_DIAGONAL_COST

            if tentative_g_score < neighbor.g:
                neighbor.parent = current_node
                neighbor.g = tentative_g_score
                neighbor.h = heuristic(neighbor, grid[GOAL[0]][GOAL[1]])
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_set.queue:
                    open_set.put(neighbor)

    return None


@app.route('/runny_nose', methods=['POST', 'GET'])
def run():
    config()

    path = astar()

    output_data = {
        "waypoints": [],
        "markers": []
    }

    if path is not None:
        output_data["waypoints"] = generate_path(path)
    else:
        print("No path found")

    with open("path_output.path", 'w') as file:
        json.dump(output_data, file, indent=2)

    file_path = 'path_output.path'

    return send_file(file_path, as_attachment=True)


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)


def gen_frames():
    global moving_obstacles
    while True:
        success, frame = camera.read()
        if not success:
            continue
        else:
            frame, coords = TFLite_detection_webcam.run_model(frame)
            for obs in moving_obstacles:
                grid[obs[0]][obs[1]].obstacle = False
            print(moving_obstacles)
            moving_obstacles = []
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            for obj in coords:
                x1 = obj.box[0]
                y1 = obj.box[1]
                x2 = obj.box[2]
                y2 = obj.box[3]
                x_1, y_1, x_2, y_2 = define_world_space_position(
                    x1, y1, x2, y2)
                x_1 = int(x_1)
                x_2 = int(x_2)
                y_1 = int(y_1)
                y_2 = int(y_2)
                moving_obstacles += construct_shape(
                    (x_1, y_1), (x_2, y_2), (x_1, y_2), (x_2, y_1))
                for obs in moving_obstacles:
                    grid[obs[0]][obs[1]].obstacle = True

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def define_world_space_position(x1, y1, x2, y2):
    global camera, fov_horizontal, fov_vertical
    # Camera specifications
    # Width resolution of the camera
    resolution_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Height resolution of the camera
    resolution_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fov_horizontal = 60  # Horizontal field of view in degrees
    fov_vertical = 40  # Vertical field of view in degrees

    # Real-world dimensions
    grid_width = 16.5
    grid_height = 8

    # Camera position on the grid
    camera_x = 82
    camera_y = 40

    # Camera rotation angle in degrees
    camera_rotation = 45  # Example angle

    # Box properties
    box_xmin = x1  # X-coordinate of the bottom-left corner of the box on the screen
    box_ymin = y1  # Y-coordinate of the bottom-left corner of the box on the screen
    box_xmax = x2  # X-coordinate of the top-right corner of the box on the screen
    box_ymax = y2  # Y-coordinate of the top-right corner of the box on the screen

    # Calculate the pixel size in real-world space
    pixel_size_horizontal = fov_horizontal / resolution_width
    pixel_size_vertical = fov_vertical / resolution_height

    # Calculate the center of the box in pixels
    box_center_x = (box_xmin + box_xmax) / 2
    box_center_y = (box_ymin + box_ymax) / 2

    # Calculate the angle offsets from the center of the camera's FOV
    angle_offset_horizontal = (
        box_center_x - resolution_width / 2) * pixel_size_horizontal
    angle_offset_vertical = (
        box_center_y - resolution_height / 2) * pixel_size_vertical

    # Convert the angle offsets to real-world distances
    distance_horizontal = math.tan(math.radians(
        angle_offset_horizontal)) * grid_width / 2
    distance_vertical = math.tan(math.radians(
        angle_offset_vertical)) * grid_height / 2

    # Adjust the angle offsets based on the camera's rotation angle
    adjusted_angle_offset_horizontal = math.radians(
        camera_rotation) + math.radians(angle_offset_horizontal)
    adjusted_angle_offset_vertical = math.radians(
        camera_rotation) + math.radians(angle_offset_vertical)

    # Convert the adjusted angle offsets to real-world distances
    adjusted_distance_horizontal = math.tan(
        adjusted_angle_offset_horizontal) * grid_width / 2
    adjusted_distance_vertical = math.tan(
        adjusted_angle_offset_vertical) * grid_height / 2

    # Calculate the box size in real-world space
    box_width_pixels = box_xmax - box_xmin
    box_height_pixels = box_ymax - box_ymin
    box_width_real = box_width_pixels * pixel_size_horizontal
    box_height_real = box_height_pixels * pixel_size_vertical

    # Calculate the coordinates of the box's corners in real-world space
    bottom_left_x = camera_x + distance_horizontal - \
        (box_width_real / 2) + adjusted_distance_horizontal
    bottom_left_y = camera_y + distance_vertical - \
        (box_height_real / 2) + adjusted_distance_vertical
    top_right_x = bottom_left_x + box_width_real
    top_right_y = bottom_left_y + box_height_real

    return (bottom_left_x, bottom_left_y, top_right_x, top_right_y)


def set_fov(fov_h, fov_v):
    global fov_horizontal, fov_vertical
    fov_horizontal = fov_h
    fov_vertical = fov_v


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    square_size = min(math.floor(800 / grid_height),
                      math.floor(1650 / grid_width))
    app.run()

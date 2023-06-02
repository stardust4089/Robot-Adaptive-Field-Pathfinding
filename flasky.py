import json
import math
from flask import Flask, render_template, request, jsonify

import os

app = Flask(__name__)

# Create an empty array to store the clicked points
obstacles = [(164,0)]

# Define the size of the grid
grid_width = 165
grid_height = 80
square_size = 0  # Size of each square in pixels

# Define the colors
color_normal = "gray"  # Color for normal squares
color_clicked = "blue"  # Color for clicked squares
color_obstacle = "red"  # Color for obstacle squares

# Variables to track the shape construction
shape_points = []  # Stores the points for shape construction

# Function to handle mouse clicks
@app.route('/')
def index():
    return render_template('index.html', obstacles=obstacles)

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


if __name__ == '__main__':
    # Calculate the square size based on the screen size and grid size
    square_size = min(math.floor(800 / grid_height), math.floor(1650 / grid_width))

    app.run()

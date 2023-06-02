import math
import tkinter as tk
from PIL import Image, ImageTk

# Create an empty array to store the clicked points
points = []
obstacles = [(69, 56), (60, 53), (70, 55), (61, 52), (47, 53), (50, 52), (65, 52), (68, 57), (71, 56), (46, 57), (63, 52), (55, 50), (55, 59), (58, 58), (47, 55), (61, 54), (52, 51), (65, 54), (66, 53), (46, 50), (57, 50), (57, 59), (63, 54), (64, 55), (67, 54), (58, 51), (55, 52), (59, 50), (53, 55), (59, 59), (47, 57), (51, 57), (65, 56), (66, 55), (46, 52), (57, 52), (48, 58), (64, 57), (67, 56), (58, 53), (45, 53), (47, 50), (56, 53), (59, 52), (55, 54), (47, 59), (61, 58), (51, 50), (51, 59), (66, 57), (68, 54), (48, 51), (46, 54), (49, 50), (49, 59), (62, 53), (53, 50), (53, 59), (70, 54), (50, 51), (61, 51), (59, 54), (47, 61), (51, 52), (52, 60), (68, 56), (46, 56), (49, 52), (49, 61), (72, 56), (73, 55), (64, 52), (53, 52), (54, 51), (55, 58), (54, 60), (70, 56), (47, 54), (60, 57), (65, 53), (66, 52), (57, 58), (46, 58), (62, 57), (64, 54), (55, 51), (54, 53), (56, 50), (45, 50), (56, 59), (47, 56), (60, 50), (51, 56), (50, 58), (52, 55), (66, 54), (46, 51), (57, 51), (48, 60), (53, 56), (63, 58), (55, 53), (45, 52), (56, 52), (54, 55), (47, 58), (69, 55), (60, 52), (50, 60), (66, 56), (68, 53), (71, 55), (62, 52), (63, 51), (64, 58), (165, 80), (47, 51), (69, 57), (60, 54), (51, 51), (61, 53), (52, 50), (51, 60), (52, 59), (72, 55), (62, 54), (53, 51), (63, 53), (54, 50), (54, 59), (67, 53), (58, 50), (58, 59), (59, 58), (52, 52), (65, 55), (64, 53), (54, 52), (56, 58), (67, 55), (58, 52), (59, 51), (60, 58), (61, 57), (50, 57), (65, 57), (46, 53), (48, 50), (57, 53), (48, 59), (62, 58), (49, 58), (63, 57), (54, 54), (56, 51), (45, 51), (67, 57), (69, 54), (60, 51), (59, 53), (50, 50), (47, 60), (50, 59), (52, 56), (68, 55), (71, 54), (62, 51), (49, 51), (46, 55), (48, 52), (48, 61), (49, 60), (53, 60)]# Define the size of the grid
grid_width = 165
grid_height = 80
square_size = 0  # Size of each square in pixels

# Define the colors
color_normal = "gray"  # Color for normal squares
color_clicked = "blue"  # Color for clicked squares
color_obstacle = "red"  # Color for obstacle squares

# Variables to track the shape construction
shape_points = []  # Stores the points for shape construction
shape_lines = []  # Stores the line IDs for shape lines
# Function to handle mouse clicks
def add_point(event):
    # Calculate the coordinates of the clicked point in the grid
    x = event.x // square_size
    y = ((event.y // square_size) % 80)
    point = (x, y)
    # Check if the point is already in the array
    if point in points or point in obstacles:
        # Remove the point from the array
        points.remove(point)

        # Change the color of the square back to normal
        canvas.itemconfig(square_ids[x][y % 80], fill="")
        print("Points:", points)
    else:
        # Add the point to the array
        points.append(point)

        # Change the color of the clicked square
        canvas.itemconfig(square_ids[x][(y % 80)], fill=color_obstacle)
        print("Points:", points)

    # Check if the Shift key is held down
    if event.state & 0x1:
        # Calculate the coordinates of the clicked point in the grid
        x = event.x // square_size
        y = ((event.y // square_size) % 80)
        point = (x, y)
        # Check if the shape construction is in progress
        if len(shape_points) < 4:
            # Add the point to the shape points
            shape_points.append(point)

            # Change the color of the clicked square
            canvas.itemconfig(square_ids[x][y % 80], fill=color_clicked)

            # If the shape points are complete, construct the shape
            if len(shape_points) == 4:
                construct_shape()
        else:
            shape_points.clear()
        

        # Print the updated array
        print("\n\n\n")
        print("Obstacles:", obstacles)

# Function to construct the shape and color the lines between the points
def construct_shape():
    global shape_points, obstacles
    # Get the coordinates of the shape points
    x1, y1 = shape_points[0]
    x2, y2 = shape_points[1]
    x3, y3 = shape_points[2]
    x4, y4 = shape_points[3]
    
    points = []

    points += bresenham_line(x1, y1, x2, y2)
    points += bresenham_line(x2, y2, x3, y3)
    points += bresenham_line(x3, y3, x4, y4)
    points += bresenham_line(x4, y4, x1, y1)
    points += bresenham_line(x1, y1, x3, y3)
    points += bresenham_line(x2, y2, x4, y4)
    """
    for point in points:
        x, y = point
        canvas.itemconfig(square_ids[x][y], fill=color_obstacle)"""
    obstacles += points
    obstacles = [*set(obstacles)]
    
    for obs in obstacles:
            x, y = obs
            canvas.itemconfig(square_ids[x][y], fill=color_obstacle)
    
    # Clear the shape points
    shape_points = []
    
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

        """
        # Check if the current point is diagonal to the previous point
        if (x0 - sx, y0 - sy) and (x0 - sx, y0) and (x0, y0 - sy):
            if abs(x0 - (x0 - sx)) == abs(y0 - (y0 - sy)):
                #if (x0 - sx, y0 - sy):
                    points.append((x0 - sx, y0 - sy))"""
        # Check if the current point is diagonal to the previous point or on a directly diagonal line
        #points.append((x0-sx, y0-sy))
        points.append((x0+1, y0))
        points.append((x0, y0+1))


    return points


# Create the main window
window = tk.Tk()

# Create a frame for the sidebar
sidebar = tk.Frame(window, width=200, bg="white")
sidebar.pack(side="left", fill="y")

# Create buttons in the sidebar
button1 = tk.Button(sidebar, text="Button 1", width=15)
button1.pack(pady=10)
button2 = tk.Button(sidebar, text="Button 2", width=15)
button2.pack(pady=10)
button3 = tk.Button(sidebar, text="Button 3", width=15)
button3.pack(pady=10)

# Calculate the screen size
screen_width = window.winfo_screenwidth() - 200
screen_height = window.winfo_screenheight()

# Calculate the square size based on the screen size and grid size
square_size = min(math.floor(screen_width / grid_width), math.floor(screen_height / grid_height))

# Calculate the canvas dimensions based on the grid size and square size
canvas_width = grid_width * square_size
canvas_height = grid_height * square_size

# Create a canvas to draw the grid
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)

# Load the background image
background_image = Image.open("C:/Users/Clementine/Desktop/field23_1_1_1650x800.png")  # Replace with the actual path to your image file

# Resize the image to fit the grid
background_image = background_image.resize((grid_width * square_size, grid_height * square_size), Image.ANTIALIAS)

# Create a PhotoImage object from the resized image
background_photo = ImageTk.PhotoImage(background_image)

# Create a canvas to draw the grid
canvas = tk.Canvas(window, width=grid_width*square_size, height=grid_height*square_size)

canvas.pack()

# Set the background image
canvas.create_image(0, 0, anchor="nw", image=background_photo)

# Set the background color
canvas.configure(bg="white")

# Bind the mouse click event to the canvas
canvas.bind("<Button-1>", add_point)

# Draw the grid on the canvas and store the square IDs
square_ids = []
for x in range(grid_width):
    row = []
    for y in range(grid_height):
        x1 = x * square_size
        y1 = y * square_size
        x2 = x1 + square_size
        y2 = y1 + square_size
        square_id = canvas.create_rectangle(x1, y1, x2, y2, outline="gray")
        row.append(square_id)
    square_ids.append(row)

# Start the main event loop
window.mainloop()

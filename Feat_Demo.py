import taichi as ti
import math
import numpy as np

ti.init(arch=ti.vulkan)

# Window setup
window_width = 1200
window_height = 400
part_width = window_width // 3
part_height = window_height

# Circle parameters
circle_center = (0, 1)
circle_radius = 1
north_pole = (circle_center[0], circle_center[1] + circle_radius)

# Time variables
t = 0.0
speed = 2

# GUI setup
gui = ti.GUI("Input Domain Representations", (window_width, window_height))

def north_pole_projection(point):
    """Project point from circle to y=0 line through north pole"""
    x, y = point
    nx, ny = north_pole
    
    if abs(y - ny) < 1e-6:  # Near north pole
        return float('inf')  # or None for no intersection
    
    # Parametric line from north pole to point
    # Solve for parameter where y=0
    t_param = -ny / (y - ny)
    proj_x = nx + t_param * (x - nx)
    
    return proj_x

def g1(t):
    return (t-1) * 0.5 + 0.1
    #return -0.5 * t**2 + 2

def g2(t):
    return -0.5 * t**2 + 2

g = g1
btn = gui.button('Switch Function')
is_btn_pressed = False
while gui.running:

    gui.clear(0xffffff)
    # Make a button to switch between functions
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.ESCAPE:
            break
        if gui.event.key == btn:
            is_btn_pressed = True
        # pause and unpause
        if gui.event.key == ' ':
            speed = 0 if speed != 0 else 1

    if is_btn_pressed:
        is_btn_pressed = False
        if g == g1:
            g = g2
        else:
            g = g1

    # Update time
    t += 0.01 * speed
    
    # Calculate moving point on circle
    circle_x = math.cos(t)
    circle_y = 1 + math.sin(t)
    circle_point = (circle_x, circle_y)
    
    # Calculate projection on y=0 line
    proj_x = north_pole_projection(circle_point)
    
    # Draw the three parts with borders
    gui.line((1/3, 0), (1/3, 1), color=0x000000, radius=2)
    gui.line((2/3, 0), (2/3, 1), color=0x000000, radius=2)
    
    # Part 1: Circle and projection
    # ---------------------------------
    # Transform coordinates to first part
    def transform1(x, y):
        scale = part_height * 0.2
        return ((x * scale + part_width * 0.5)/window_width, 
                (y * scale + part_height * 0.5)/window_height)
    
    # Draw circle
    circle_pos = transform1(*circle_center)
    gui.circle(circle_pos, radius=circle_radius * part_height * 0.2, color=0xF0F0FF)
    
    # Draw north pole
    np_pos = transform1(*north_pole)
    gui.circle(np_pos, radius=5, color=0xFF0000)
    
    # Draw moving point
    moving_pos = transform1(circle_x, circle_y)
    gui.circle(moving_pos, radius=5, color=0x0000FF)
    
    # Draw projection line
    if abs(circle_y - north_pole[1]) > 1e-3:  # Not at north pole
        
        # Draw projected point if finite
        if abs(proj_x-circle_pos[0]) < 100:#part_width*0.5/window_width:  # Practical infinity for visualization
            proj_pos = transform1(proj_x, 0)
            gui.circle(proj_pos, radius=5, color=0xFF00FF)
            gui.line(np_pos, proj_pos, color=0x888888)
    
    # Draw y=0 line
    left = transform1(-2, 0)
    right = transform1(2, 0)
    gui.line(left, right, color=0x000000)
    
    # Part 2: Function of t
    # ---------------------------------
    def transform2(t_val, g_val):
        x = (t_val + 3) / 6  # Map t from -3 to 3 to 0-1
        y = (g_val + 2) / 4  # Map g from -2 to 2 to 0-1
        return ( (x * part_width * 0.8 + part_width * 1.1) / window_width,
                 (y * part_height * 0.8 + part_height * 0.1) / window_height )
    
    # Draw axes
    origin = transform2(0, 0)
    x_end = transform2(3, 0)
    y_end = transform2(0, 2)
    gui.line(origin, x_end, color=0x000000)
    gui.line(origin, y_end, color=0x000000)
    
    prev_point = None
    for t_val in np.linspace(-3.14, 3.14, 100):
        g_val = g(t_val)
        point = transform2(t_val, g_val)
        if prev_point:
            gui.line(prev_point, point, color=0x8080FF)
        prev_point = point
    
    # Draw moving point
    tc = (t+math.pi)%(math.pi*2) - math.pi
    current_g1 = g(tc)
    moving_pos_g1 = transform2(tc, current_g1)
    gui.circle(moving_pos_g1, radius=5, color=0x0000FF)
    
    # Part 3: Function of projected x
    # ---------------------------------
    def transform3(x_val, g_val):
        x = (x_val + 3) / 6  # Map x from -3 to 3 to 0-1
        y = (g_val + 2) / 4  # Map g from -2 to 2 to 0-1
        return ( (x * part_width * 0.8 + part_width * 2.1)/window_width,
                 (y * part_height * 0.8 + part_height * 0.1) / window_height )
    
    # Draw axes
    origin = transform3(0, 0)
    x_end = transform3(3, 0)
    y_end = transform3(0, 2)
    gui.line(origin, x_end, color=0x000000)
    gui.line(origin, y_end, color=0x000000)
    
    # Draw function g2
    prev_point = None
    for t_val in np.linspace(-3.14, 3.14, 100):
        cir_x_val = math.cos(t_val)
        cir_y_val = 1 + math.sin(t_val)
        cir_point = (cir_x_val, cir_y_val)
        this_proj_x = north_pole_projection(cir_point)

        g_val = g(t_val)
        point = transform3(this_proj_x, g_val)

        if prev_point:
            if abs(this_proj_x) < 3:
                gui.line(prev_point, point, color=0xAA00AA)
        prev_point = point
    
    # Draw moving point if projection is finite
    if abs(proj_x) < 3:  # Only show if within visible range
        moving_pos_g2 = transform3(proj_x, current_g1)
        gui.circle(moving_pos_g2, radius=5, color=0xFF00FF)
    
    # Add labels
    gui.text("Circle and Projection", (0.15, 0.9), font_size=20, color=0x000000)
    gui.text("g(t)", (0.5, 0.9), font_size=20, color=0x000000)
    gui.text("f(x)", (0.85, 0.9), font_size=20, color=0x000000)
    
    gui.show()
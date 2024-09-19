import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

class MeshGUI:
    def __init__(self, 
        name="MeshGen Main", window_width=1200, window_height=800):
        # GGUI -- not working properly at the moment
        self.window = ti.ui.Window(
            name=name, 
            res=(window_width, window_height), pos=(15, 50), fps_limit=100)
        self.canvas = self.window.get_canvas()
        # components in a gui loop
        # 1 objects state maintain
        self.state_callbacks = []
        
        # 2 main elements display
        self.viz_callbacks = []

        # 3 key-press event handlers
        self.event_key_callbacks = {}
        self.state_data = {}
        self.state_data["nodes"] = {}
        self.state_data["edges"] = {}
        self.state_data["menu_items"] = {}
        self.state_data["file_menu_items"] = {}
        self.state_data["user_inputs"] = {
            "mouse_click": {
                "screen_pos": [0, 0],
                "to_process": False
            }
        }
        self.create_menu_ggui()

    def register_state_callback(self, func, kwargs):
        assert callable(func), "Call back functions must be callable"
        self.state_callbacks.append((func, kwargs))

    def register_viz_callbacks(self, func, kwargs):
        assert callable(func), "Call back functions must be callable"
        print("add call back")
        self.viz_callbacks.append((func, kwargs))

    def register_event_callbacks(self, func, event_key):
        assert callable(func), "Call back functions must be callable"
        self.event_key_callbacks[event_key] = func

    def create_menu_ggui(self):
        # == flags ==
        # todo: use radio check items to handle mutual exclusion
        # (note it may be difficult for actions to de-select a "preparation" check)
        MI = self.state_data['menu_items']
        MI["item1"] = {
            "type":"toggle_flag",
            "text":"Option-1",
            "val":False,
            "interaction":None,
            "exclude":[],
            "viz_func": lambda x:None
        }

    def reset_state(self):
        pass
        # V, F, msk = reset()
        # lon_min, lat_min = V.min(axis=0)
        # lon_max, lat_max = V.max(axis=0)

    def main_loop(self):
        wnd = self.window
        mmenu = self.state_data['menu_items']
        fmenu = self.state_data['file_menu_items']
        userin = self.state_data['user_inputs']
        dt = 1
        time_elapsed = 0
        while wnd.running:
            gui = wnd.get_gui()
            # Display menu items
            with gui.sub_window("Menu", x=0.01, y=0.01, width=0.4, height=0.3):  
                for k, mitem in mmenu.items(): 
                    if mitem['type'] == 'toggle_flag':
                        mitem['interaction'] = gui.button(
                            f"{mitem['text']}:{'ON' if mitem['val'] else 'OFF'}")
                    if mitem['type'] == 'action' and mitem['is_valid'](self.state_data):
                        mitem['interaction'] = gui.button(f"{mitem['text']}")

            if wnd.is_pressed(ti.ui.LMB):
                mouse_pos_x, mouse_pos_y = wnd.get_cursor_pos()
                userin['mouse_click']['screen_pos'] = [mouse_pos_x, mouse_pos_y]
                userin['mouse_click']['to_process'] = True
            
            for k, mitem in fmenu.items():
                if mitem['type'] == 'action' and mitem['interaction']:
                    mitem['action'](self.state_data)

            # Topology operation menu maintain
            # - handle user instructions by setting up states for further proc
            #   in self.viz_callbacks
            for k, mitem in mmenu.items():
                if mitem['type'] == 'toggle_flag':
                    if mitem['interaction']:
                        mitem['val'] = not mitem['val']
                        if mitem['val']:
                            for k in mitem['exclude']:
                                self.menu_items[k]['val'] = False

                if mitem['type'] == 'action':
                    if mitem['interaction']:
                        # print(f"Activating action item {mitem}")
                        mitem['action'](self.state_data)
                        for k in mitem['exclude']:
                            self.menu_items[k]['val'] = False
                        mitem['interaction'] = False
                
            if time_elapsed > dt:
                for f, a in self.state_callbacks:
                    f(self.state_data, **a) # each entry (func, args)
                time_elapsed -= dt

            for f, a in self.viz_callbacks:
                f(self, **a) # visualiser need a canvas to draw on


            wnd.show()
            time_elapsed += 0.1
            

################################################################################
# Variables
################################################################################

# Vars for n x n neuron states
n = 64
V = ti.field(ti.f32, shape=(n*n,)) # the activation states of the neurons
H = ti.field(ti.f32, shape=(n*n,)) # the potential feeled by the neurons
positions = ti.Vector.field(2, dtype=ti.f32, shape=(n*n,)) # 2D locations in (0,1)
colors = ti.Vector.field(3, dtype=ti.f32, shape=(n*n,))
T = 10
B = 1
# Assume a maximum of 2 * n * n edges (each vertex except boundaries has 4 edges)
max_edges = 4 * n * n
edges = ti.Vector.field(2, dtype=ti.i32, shape=(max_edges,))  # Each edge connects two vertices
W = ti.field(dtype=ti.f32, shape=(max_edges,))  # Each edge has a weight
edge_count = ti.field(dtype=ti.i32, shape=())  # Counter for the number of edges


################################################################################
# Kernels
################################################################################

# Initialize the vertices
@ti.kernel
def initialize_neurons():
    for i in range(n * n):
        y = float(i // n) / n
        x = float(i % n) / n
        positions[i] = ti.Vector([x, y])
        if ti.random(ti.f32) >= 0.5:
            V[i] = 1.0
        else:
            V[i] = -1.0
        


# Generate edges based on the vertex grid
@ti.kernel
def initialize_edges():
    idx = 0
    for n0 in range(n*n):
        i0 = n0 // n
        j0 = n0 % n
        if j0 > 0: # we can connect to the left
            n1 = i0 * n + (j0 - 1)
            edges[idx] = ti.Vector([n0, n1])
            idx += 1
        if j0 < n-1: # we can connect to the left
            n1 = i0 * n + (j0 + 1)
            edges[idx] = ti.Vector([n0, n1])
            idx += 1
        if i0 > 0: # we can connect to the left
            n1 = (i0 - 1) * n + j0
            edges[idx] = ti.Vector([n0, n1])
            idx += 1
        if i0 < n-1: # we can connect to the left
            n1 = (i0 + 1) * n + j0
            edges[idx] = ti.Vector([n0, n1])
            idx += 1
    edge_count[None] = idx  # Store the total number of edges

@ti.kernel
def initialize_weights():
    ne = edge_count[None]
    for i in range(ne):
        W[i] = 1.0

@ti.kernel
def update_H():
    for i in range(n*n):
        H[i] = B
    ne = edge_count[None]
    for i in range(ne):
        rec = int(edges[i][0]) # receiving
        src = int(edges[i][1]) # sending
        H[rec] += W[i] * V[src]
        
@ti.kernel
def update_V():
    for i in range(n*n):
        threshold = 1 / (1 + ti.exp(- 2* H[i] / T))
        if ti.random(ti.f32) <= threshold:
            V[i] = 1
        else:
            V[i] = -1

@ti.kernel
def update_colors():
    for i in range(n*n):
        if V[i] > 0.0:
            colors[i] = ti.Vector([1.0, 0.0, 0.0])
        else:
            colors[i] = ti.Vector([0.0, 0.0, 1.0])


initialize_neurons()
initialize_edges()
initialize_weights()

def draw_mesh(gui):
    # Access vertices and edges from the state data
    pos = gui.state_data['nodes']
    edges = gui.state_data['edges']

    # n_vertices = vertices.shape[0]    # Total number of vertices
    # n_edges = gui.state_data['edge_count'][None]  # Total number of edges
    
    # Extract positions of vertices for drawing
    # positions = vertices.to_numpy() / 64  # Normalizing to [0, 1] for canvas display
    
    # Draw the vertices as points on the canvas
    gui.canvas.circles(pos, radius=0.002, per_vertex_color=colors)

def update(s):
    """
    Args
    s: state dictionary, contains
      'nodes', 'edges'
    """
    update_H()
    update_V()
    update_colors()

main_window = MeshGUI()
main_window.state_data['nodes'] = positions
main_window.state_data['edges'] = edges
main_window.register_state_callback(update, {})
main_window.register_viz_callbacks(draw_mesh, {})
main_window.main_loop()


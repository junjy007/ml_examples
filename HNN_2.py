import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)
J = np.load("hnn_weight.npy")
N = 64 * 64
J = J.astype(np.float32) / N 
assert J.shape[0] ==  J.shape[1] and J.shape[0] == N


################################################################################
# Variables
################################################################################

# Vars for n x n neuron states
n = 64
V = ti.field(ti.f32, shape=(n*n,)) # the activation states of the neurons
fixed_V = ti.field(ti.f32, shape=(n*n,))
H = ti.field(ti.f32, shape=(n*n,)) # the potential feeled by the neurons
positions = ti.Vector.field(2, dtype=ti.f32, shape=(n*n,)) # 2D locations in (0,1)
colors = ti.Vector.field(3, dtype=ti.f32, shape=(n*n,))
T = ti.field(ti.f32, shape=())
B = ti.field(ti.f32, shape=())
# Assume a maximum of 2 * n * n edges (each vertex except boundaries has 4 edges)
# edges = ti.Vector.field(2, dtype=ti.i32, shape=(max_edges,))  # Each edge connects two vertices
W = ti.field(dtype=ti.f32, shape=(N, N))  # Each edge has a weight
W.from_numpy(J)

################################################################################
# Kernels
################################################################################

# Initialize the vertices
@ti.kernel
def init():
    T[None] = 1
    B[None] = 0
    for i in range(n*n):
        fixed_V[i] = -999

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
# @ti.kernel
# def initialize_edges():
#     idx = 0
#     for n0 in range(n*n):
#         i0 = n0 // n
#         j0 = n0 % n
#         if j0 > 0: # we can connect to the left
#             n1 = i0 * n + (j0 - 1)
#             edges[idx] = ti.Vector([n0, n1])
#             idx += 1
#         if j0 < n-1: # we can connect to the left
#             n1 = i0 * n + (j0 + 1)
#             edges[idx] = ti.Vector([n0, n1])
#             idx += 1
#         if i0 > 0: # we can connect to the left
#             n1 = (i0 - 1) * n + j0
#             edges[idx] = ti.Vector([n0, n1])
#             idx += 1
#         if i0 < n-1: # we can connect to the left
#             n1 = (i0 + 1) * n + j0
#             edges[idx] = ti.Vector([n0, n1])
#             idx += 1
#     edge_count[None] = idx  # Store the total number of edges

@ti.kernel
def update_H():
    for i in range(N):
        H[i] = B[None]
        for j in range(N):
            H[i] += W[i, j] * V[j]
        
@ti.kernel
def update_V():
    for i in range(n*n):
        if fixed_V[i] > -10:
            V[i] = fixed_V[i]
        else:

            threshold = 1 / (1 + ti.exp(- 2* H[i] / T[None]))
            if ti.random(ti.f32) <= threshold:
                V[i] = 1.0
            else:
                V[i] = -1.0

@ti.kernel
def update_colors():
    for i in range(n*n):
        if V[i] > 0.1:
            colors[i] = ti.Vector([1.0, 1.0, 1.0])
        else:
            colors[i] = ti.Vector([.0, 0.0, 0.0])


def update():
    """
    Args
    s: state dictionary, contains
      'nodes', 'edges'
    """
    update_H()
    update_V()
    update_colors()

@ti.kernel
def set_pos(i:int):
    fixed_V[i] = 1.0
@ti.kernel
def set_neg(i:int):
    fixed_V[i] = -1.0

window = ti.ui.Window(
    name="Model Out", res=(512, 512), pos=(15, 50), fps_limit=100)

init()
initialize_neurons()
while window.running:
    gui = window.get_gui()
    canvas = window.get_canvas()
    T[None] = ti.exp(gui.slider_float("Temperature", ti.log(T[None]), minimum=-10, maximum=5))
    B[None] = gui.slider_float("ExField", B[None], minimum=-10, maximum=10)
    do_reset = gui.button("reset")
    if do_reset:
        init()

    pos = window.get_cursor_pos()
    xi = min(int(pos[0] * n + 0.5), n-1)
    yi = min(int(pos[1] * n + 0.5), n-1)
    print(xi, yi)
    if window.get_event():
        if window.event.key == ti.ui.LMB:
            set_pos(yi*n+xi)
        elif window.event.key == ti.ui.RMB:
            set_neg(yi*n+xi)
        

    update()
    canvas.circles(positions, radius=0.003, per_vertex_color=colors)
    window.show()



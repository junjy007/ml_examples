import taichi as ti
import numpy as np
from sklearn.datasets import load_iris

ti.init(arch=ti.cpu) 

# Define the resolution of the image
resolution = 512

# Create a Taichi field to store the image
image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution, resolution))
N = 100
iris_db = load_iris()
X_np = iris_db['data'][:N, :2]
Y_np = iris_db['target'][:N].astype(np.int32)
X_np = (X_np - X_np.min(axis=0))
X_np = X_np / X_np.max(axis=0) * 1.8
X_np -= 0.9
X_np = X_np.astype(np.float32)

WBND = 3.0
XBND = 1.


XSELE_TH = 0.1
WN = 1000
XN = 1000
MINLOGP = -6
vmin = ti.field(ti.f32, shape=())
vmax = ti.field(ti.f32, shape=())
W_curr = ti.Vector.field(2, ti.f32, shape=())
b = ti.field(ti.f32, shape=())
act = ti.field(ti.i32, shape=())

# ... for managing X
X = ti.Vector.field(2, ti.f32, shape=(N,))
Y = ti.field(ti.i32, shape=(N,))
X.from_numpy(X_np)
Y.from_numpy(Y_np)
click_in_x = ti.Vector.field(2, ti.f32, shape=(1,))
xsele_i = ti.field(ti.i32, shape=())

# flags
flag_select_w = ti.field(ti.i32, shape=())
flag_select_x = ti.field(ti.i32, shape=()) # 0 invalid, 1 add 2 remove
train_errors = ti.field(ti.i32, shape=(N,))

@ti.kernel
def init():
    vmin[None] = -1.0
    vmax[None] = +1.0
    b[None] = 0
    act[None] = 0
    flag_select_w[None] = 0
    flag_select_x[None] = 0
    for i, j in image:
        image[i, j] = ti.Vector([.5, .5, .5])

    W_curr[None][0] = 0.5 # ti.randn() * 0.5
    W_curr[None][1] = 0.3 # ti.randn() * 0.5
    xsele_i[None] = -1

@ti.func
def cmap(s:float) -> ti.Vector:
    c = ti.Vector([0.0, 0.0, 0.0])
    if s < vmin[None]:
        c[2] = 1.0
    elif s > vmax[None]:
        c[0] = 1.0
    else:
        s1 = (s-vmin[None]) / (vmax[None] - vmin[None])
        c[0] = s1
        c[2] = 1.0-s1
    return c

@ti.func
def actf(s:ti.f32) -> ti.f32:
    if s > 0:
        s = 1.0
    else:
        s = -1.0
    return s

@ti.kernel
def compute_image():
    for i, j in image:
        x0 = (float(i) / resolution) * (2*XBND) - XBND
        x1 = (float(j) / resolution) * (2*XBND) - XBND
        s = W_curr[None][0] * x0 + W_curr[None][1] * x1
        if act[None]:
            s = actf(s)
        image[i, j] = cmap(s)

@ti.kernel
def update_train_errors():
    for i in range(N):
        s = W_curr[None][0] * X[i][0] + W_curr[None][1] * X[i][1]
        if s > 0:
            if Y[i] > 0:
                train_errors[i] = 0
            else:
                train_errors[i] = 1
        else:
            if Y[i] > 0:
                train_errors[i] = 1
            else:
                train_errors[i] = 0

def W_to_screen_pos(W):
    return ti.Vector([
        (W[0]) / (2.0 * WBND), 
        (W[1]) / (2.0 * WBND) ])

@ti.kernel
def select_x():
    d = 999.0
    j = -1
    for i in range(N):
        di = ti.math.length(X[i] - click_in_x[0])
        if di < d and train_errors[i] > 0:
            d = di
            j = i
            print(di, j)
    if d < XSELE_TH:
        xsele_i[None] = j
        print(j, "selected")
    else:
        xsele_i[None] = -1
    
init()
# Create a GUI window
gui = ti.GUI(
    name="Model Output", res=(resolution, resolution))
is_update_ready = False
does_update_w = gui.button('Update W')
does_toggle_act = gui.button('Toggle Act')
w_update = ti.Vector([0., 0.])
while gui.running:
    compute_image()
    update_train_errors()
    mouse = gui.get_cursor_pos()
    for e in gui.get_events(gui.RELEASE):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == 'u':
            if is_update_ready:
                W_curr[None] += w_update
                print("New W", W_curr)
                is_update_ready = False
        elif e.key == 'a':
            act[None] = 1 - act[None]
        elif e.key == gui.LMB:
            click_in_x[0][0] = (mouse[0] - 0.5) * 2.0
            click_in_x[0][1] = (mouse[1] - 0.5) * 2.0
            # Find a sample, 1. wrongly classified and 2. clicked
            select_x()
            if xsele_i[None] >= 0:
                is_update_ready = True
        
            
    gui.set_image(image)
    gui.arrow(orig=[0.5, 0.5], direction=W_to_screen_pos(W_curr[None]),
              radius=2, color=0xFF8040)
    gui.circles(pos=X_np / 2.0 + np.array([0.5, 0.5]), radius=7,
                palette=[0x4040FF, 0xFF4040], palette_indices=Y_np)
    gui.circles(pos=X_np / 2.0 + np.array([0.5, 0.5]), radius=3,
                palette=[0x000000, 0xFFFF00], palette_indices=train_errors)
    
    if is_update_ready:
        i = xsele_i[None]
        if Y[i] > 0:
            w_update = ti.Vector(X_np[i])
        else:
            w_update = ti.Vector(-X_np[i])
        gui.arrow(orig=[0.5, 0.5], direction=w_update / 2.0,
                  radius=2, color=0x00A020)
        
    gui.show()

    

    
    

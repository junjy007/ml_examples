import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

m = 1.0
n = 1
nv = 1
x = ti.Vector.field(2, ti.f32, shape=(n,))
v = ti.Vector.field(2, ti.f32, shape=(n,))
H = ti.field(dtype=ti.f32, shape=(n,))  # Counter for the number of edges
update_flags = ti.field(dtype=ti.i32, shape=(n,))
xvis = ti.Vector.field(2, ti.f32, shape=(nv,))
phvis = ti.Vector.field(2, ti.f32, shape=(nv,))
dt = 0.01

@ti.kernel
def init():
    for i in range(n):
        H[i] = 1.0
        update_flags[i] = 0
        x[i] = ti.Vector([0, 0])
        # 2 * v^2 * m = E
        v[i] = ti.Vector([ti.sqrt(2.0*H[i]/m), 0])

@ti.kernel
def update():
    for i in range(n):
        x[i] += dt * v[i]
        v[i] += - dt * x[i]
        if update_flags[i] and x[i][0] ** 2 < H[i]:
            # update velocity for changed 
            update_flags[i] = 0
            E_new = float(H[i] - x[i][0] ** 2)
            print(E_new)
            if v[i][0] >= 0:
                v[i] = ti.Vector([ti.sqrt(2.0*E_new/m), 0])
            else:
                v[i] = ti.Vector([-ti.sqrt(2.0*E_new/m), 0])

@ti.kernel
def update_visual():
    for i in range(nv):
        xvis[i][0] = (x[i][0] + 2.0) / 10.0
        xvis[i][1] = 0.5

        phvis[i][0] = (x[i][0])/10 + 0.75
        phvis[i][1] = (v[i][0])/10 + 0.5


window = ti.ui.Window(
    name="demo", 
    res=(800, 600), pos=(15, 50), fps_limit=100)

init()
H_val = H[0]
while window.running:
    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.8, 0.8, 0.8))
    H_val_old = H_val
    H_val = gui.slider_float("Energy", H_val, minimum=0.1, maximum=3)
    if H_val != H_val_old:
        H[0] = H_val
        update_flags[0] = 1

    update()
    update_visual()

    canvas.circles(xvis, radius=0.01, color=(0.0, 0.0, 0.5))
    canvas.circles(phvis, radius=0.008, color=(1.0, 0.0, 0.0))
    window.show()


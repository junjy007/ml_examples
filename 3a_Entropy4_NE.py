import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

dt = 0.001
m = 1.0
maxN = 10000
ymin, ymax = 0.1, 0.9
xmin, xmax = 0.1, 0.9
damping = 0.99
threshold = 0.016
collision_k = 2000
N = ti.field(ti.i32, shape=())
E = ti.field(ti.f32, shape=())
hot_e = ti.field(ti.f32, shape=())
cool_e = ti.field(ti.f32, shape=())
entropy = ti.field(ti.f32, shape=())

q0 = ti.Vector.field(2, ti.f32, shape=(maxN,))
v0 = ti.Vector.field(2, ti.f32, shape=(maxN,))
f0 = ti.Vector.field(2, ti.f32, shape=(maxN,))

walls = ti.Vector.field(2, ti.f32, shape=(8,))

q0_vis = ti.Vector.field(2, ti.f32, shape=(maxN,))

@ti.kernel
def init():
    N[None] = 8000
    hot_e[None] = 1.0
    cool_e[None] = 1.0

    walls[0] = ti.Vector([xmax, ymin])
    walls[1] = ti.Vector([xmin, ymin])

    walls[2] = ti.Vector([xmin, ymin])
    walls[3] = ti.Vector([xmin, ymax])

    walls[4] = ti.Vector([xmin, ymax])
    walls[5] = ti.Vector([xmax, ymax])

    walls[6] = ti.Vector([xmax, ymin])
    walls[7] = ti.Vector([xmax, ymax])

    for i in range(N[None]):
        q0[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        # 2 * v^2 * m = E
        v0[i] = ti.Vector([ti.random(ti.f32) - 0.5, ti.random(ti.f32) - 0.5]) * 2
        E[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5


@ti.kernel
def update():
    E[None] = 0
    # Classical mechanics
    for i in range(N[None]):
        # f0[i] = ti.Vector([0, 0])
        for j in range(i+1, N[None]):
            if ti.math.length(q0[i] - q0[j]) < threshold:
                tmp = v0[i][0]
                v0[i][0] = v0[j][0]
                v0[j][0] = tmp

                tmp = v0[i][1]
                v0[i][1] = v0[j][1]
                v0[j][1] = tmp
                # f0[i] += (q0[i] - q0[j]) * collision_k


    for i in range(N[None]):
        # Wall boundaries
        if q0[i][1] < 0:
            q0[i][1] = 0
            # f0[i][1] += collision_k
            if v0[i][1] < 0: 
                v0[i][1] = - v0[i][1]
        if q0[i][1] > 1:
            q0[i][1] = 1
            # f0[i][1] -= collision_k
            if v0[i][1] > 0: 
                v0[i][1] = - v0[i][1]


    for i in range(N[None]):
        q0[i] += dt * v0[i]
        # energy exchaning

    for i in range(N[None]):
        v0[i] = v0[i] + dt * f0[i]

        if q0[i][0] > 1:
            q0[i][0] = 1
            if v0[i][0] > 0: 
                v0[i][0] = - v0[i][0]
                # exchange energy
                this_e = (v0[i][0]*v0[i][0] + v0[i][1] * v0[i][1]) * m / 2
                avg_e = 0.1*this_e + 0.9*cool_e[None]
                k = ti.sqrt( avg_e ) / ti.sqrt( this_e )
                v0[i] = v0[i] * k

        if q0[i][0] < 0:
            q0[i][0] = 0
            if v0[i][0] < 0: 
                v0[i][0] = - v0[i][0]
                # exchange energy
                this_e = (v0[i][0]*v0[i][0] + v0[i][1] * v0[i][1]) * m / 2
                avg_e = 0.1*this_e + 0.9*hot_e[None]
                k = ti.sqrt( avg_e ) / ti.sqrt( this_e )
                v0[i] = v0[i] * k
        E[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5
        
@ti.kernel
def heat_up():
    hot_e[None] *= 1.1

@ti.kernel
def cool_down():
    cool_e[None] *= 0.8

@ti.kernel
def update_visual():
    for i in range(maxN):
        q0_vis[i] = ti.Vector([-1, -1])
        
    for i in range(N[None]):
        q0_vis[i][0] = q0[i][0] * (xmax - xmin) + xmin
        q0_vis[i][1] = q0[i][1] * (ymax - ymin) + ymin


window = ti.ui.Window(
    name="demo", 
    res=(800, 800), pos=(15, 50), fps_limit=100)

init()
while window.running:
    # Compute the entropy
    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.8, 0.8, 0.8))
    hotter = gui.button("Heat Up")
    cooler = gui.button("Cool Down")
    gui.text(f"hot: {hot_e[None]:.4f}, cool: {cool_e[None]:.4f} e: {E[None]/N[None]:.4f}")

    update()
    update_visual()
    
    if hotter:
        heat_up()
    if cooler:
        cool_down()

    canvas.circles(q0_vis, radius=threshold/4, color=(0.0, 0.0, 0.5))
    canvas.lines(walls, width=.01)

    window.show()


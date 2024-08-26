import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

dt = 0.005
m = 1.0
maxN = 10000
ymin, ymax = 0.1, 0.3
xmin, xmax = 0.1, 0.9
damping = 0.99

N = ti.field(ti.i32, shape=())
H = ti.field(ti.f32, shape=())
E0 = ti.field(ti.f32, shape=())
E1 = ti.field(ti.f32, shape=())
x = ti.Vector.field(2, ti.f32, shape=(maxN,))
v = ti.Vector.field(2, ti.f32, shape=(maxN,))
lid = ti.field(ti.f32, shape=())
lid_v = ti.field(ti.f32, shape=())
lid_m = 1000.0
lid_f = ti.field(ti.f32, shape=())

walls = ti.Vector.field(2, ti.f32, shape=(6,))
xvis = ti.Vector.field(2, ti.f32, shape=(maxN,))
lid_vis = ti.Vector.field(2, ti.f32, shape=(2,))

ph_vis_centre = ti.Vector([0.6, 0.6])
ph_vis_runit = 0.2 # unit physical quantity in screen length
ph_vis_q = ti.Vector.field(2, ti.f32, shape=(maxN*2,))
ph_vis_p = ti.Vector.field(2, ti.f32, shape=(maxN*2,))
ph_vis_p_clr = ti.Vector.field(3, ti.f32, shape=(maxN*2,))


@ti.kernel
def init():
    walls[0] = ti.Vector([xmax, ymin])
    walls[1] = ti.Vector([xmin, ymin])

    walls[2] = ti.Vector([xmin, ymin])
    walls[3] = ti.Vector([xmin, ymax])

    walls[4] = ti.Vector([xmin, ymax])
    walls[5] = ti.Vector([xmax, ymax])

    lid[None] = 1.0
    lid_v[None] = 0.0
    lid_f[None] = 300.0

    H[None] = 0.0
    for i in range(N[None]):
        x[i] = ti.Vector([ti.random(ti.f32) * .8, ti.random(ti.f32)])
        # 2 * v^2 * m = E
        v[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        H[None] += m * (v[i][0] * v[i][0] + v[i][1] * v[i][1]) * 0.5

    H[None] = H[None] \
        + lid[None] * lid_f[None] \
        + lid_v[None] * lid_v[None] * lid_m * .5

@ti.kernel
def reset_H():
    # when balance changed, reset H to be preserved
    H[None] = 0.0
    for i in range(N[None]):
        H[None] += m * (v[i][0] * v[i][0] + v[i][1] * v[i][1]) * 0.5

    H[None] = H[None] \
        + lid[None] * lid_f[None] \
        + lid_v[None] * lid_v[None] * lid_m * .5

@ti.kernel
def update():
    lid_v[None] -= dt * lid_f[None] / lid_m
    lid_v[None] *= damping
    lid[None] += dt * lid_v[None]

    E0[None] = H[None] - (
        lid[None] * lid_f[None] + lid_v[None] * lid_v[None] * lid_m * .5)

    E1[None] = 0.0
    for i in range(N[None]):
        x[i] += dt * v[i]
        if x[i][0] < 0:
            if v[i][0] < 0: 
                v[i][0] = - v[i][0]
        if x[i][0] > lid[None]:
            x[i][0] = lid[None]
            if v[i][0] > 0: 
                v0 = v[i][0]
                v[i][0] = lid_v[None] - v[i][0]
                lid_v[None] += m / lid_m * (v0 - v[i][0]) # conserve of momentum
        if x[i][1] < 0:
            if v[i][1] < 0: 
                v[i][1] = - v[i][1]
        if x[i][1] > 1:
            if v[i][1] > 0: 
                v[i][1] = - v[i][1]
        
        E1[None] += m * (v[i][0] * v[i][0] + v[i][1] * v[i][1]) * 0.5 

    for i in range(N[None]):
        v[i] = v[i] * ti.sqrt(E0[None] / E1[None])

    #print(E0[None])

@ti.kernel
def update_visual():
    for i in range(maxN):
        xvis[i] = ti.Vector([-1, -1])
    for i in range(N[None]):
        xvis[i][0] = x[i][0] * (xmax - xmin) + xmin
        xvis[i][1] = x[i][1] * (ymax - ymin) + ymin

    lid_vis[0] = ti.Vector([lid[None] * (xmax-xmin) + xmin, ymax])
    lid_vis[1] = ti.Vector([lid[None] * (xmax-xmin) + xmin, ymin])

    for i in range(N[None]):
        th = np.pi / 2.0 - np.pi * 2.0 / N[None] * i
        
        ph_vis_q[i*2] = ti.Vector([
            ti.cos(th) * ph_vis_runit * lid[None] + ph_vis_centre[0], 
            ti.sin(th) * ph_vis_runit * lid[None] + ph_vis_centre[1]])
        
        ph_vis_q[i*2+1] = ti.Vector([
            ti.cos(th) * ph_vis_runit * (lid[None] - x[i][0]) + ph_vis_centre[0], 
            ti.sin(th) * ph_vis_runit * (lid[None] - x[i][0]) + ph_vis_centre[1]])

        ph_vis_p[i*2] = ti.Vector([
            ti.cos(th) * ph_vis_runit * lid[None] + ph_vis_centre[0], 
            ti.sin(th) * ph_vis_runit * lid[None] + ph_vis_centre[1]])

        ph_vis_p[i*2+1] = ti.Vector([
            ti.cos(th) * ph_vis_runit * (lid[None] + abs(v[i][0])/3) + ph_vis_centre[0], 
            ti.sin(th) * ph_vis_runit * (lid[None] + abs(v[i][0])/3) + ph_vis_centre[1]])

        if v[i][0] >= 0:
            ph_vis_p_clr[i*2] = ti.Vector([1.0, 0.3, 0.0])
            ph_vis_p_clr[i*2+1] = ti.Vector([1.0, 0.3, 0.0])
        else:
            ph_vis_p_clr[i*2] = ti.Vector([0.0, 0.5, 0.0])
            ph_vis_p_clr[i*2+1] = ti.Vector([0.0, 0.5, 0.0])


window = ti.ui.Window(
    name="demo", 
    res=(800, 800), pos=(15, 50), fps_limit=100)

N[None] = 100
init()
while window.running:
    vo = lid[None]/N[None]
    e = E0[None]/N[None]
    s = ti.log(vo*ti.sqrt(2*e*m))

    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.8, 0.8, 0.8))
    old_f = lid_f[None]
    lid_f[None] = gui.slider_float("force", lid_f[None], 100, 900)
    if old_f != lid_f[None]:
        reset_H()
    gui.text(f"e={e:.3f} V={vo*N[None]:.3f} s={s:.4f}", (0, 0.8, 0, 0.5))

    update()
    update_visual()

    canvas.circles(xvis, radius=0.005, color=(0.0, 0.0, 0.5))
    canvas.lines(walls, width=.02)
    canvas.lines(lid_vis, width=.03, color=(0.6, 0, 0))
    canvas.lines(ph_vis_q, width=.002, color=(0, 0, 0.5))
    canvas.lines(ph_vis_p, width=.006, per_vertex_color=ph_vis_p_clr)
    

    # print(lid)
    # print(lid_v)
    # print(lid_vis)
    
    window.show()
    #time.sleep(1)


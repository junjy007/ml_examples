import taichi as ti
import numpy as np
import time

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

dt = 0.002
m = 1.0
maxN = 10000
ymin, ymax = 0.1, 0.3
xmin, xmax = 0.1, 0.9
damping = 0.99
threshold = 0.02
N = ti.field(ti.i32, shape=())
H0 = ti.field(ti.f32, shape=())
H1 = ti.field(ti.f32, shape=())
E0 = ti.field(ti.f32, shape=())
E1 = ti.field(ti.f32, shape=())
E = ti.field(ti.f32, shape=())
entropy = ti.field(ti.f32, shape=())

q0 = ti.Vector.field(2, ti.f32, shape=(maxN,))
q1 = ti.Vector.field(2, ti.f32, shape=(maxN,))
v0 = ti.Vector.field(2, ti.f32, shape=(maxN,))
v1 = ti.Vector.field(2, ti.f32, shape=(maxN,))

lid_q = ti.field(ti.f32, shape=())
lid_v = ti.field(ti.f32, shape=())
lid_e = ti.field(ti.f32, shape=())
lid_m = 1000.0

walls = ti.Vector.field(2, ti.f32, shape=(8,))

q0_vis = ti.Vector.field(2, ti.f32, shape=(maxN,))
q1_vis = ti.Vector.field(2, ti.f32, shape=(maxN,))
lid_vis = ti.Vector.field(2, ti.f32, shape=(2,))
vis_t = 100
entropy_vis = ti.Vector.field(2, ti.f32, shape=(vis_t*2,))
entropy_vis_update_counter = ti.field(ti.i32, shape=())
entropy_vis_update_every = 10
entropy_x0 = 0.3
entropy_y0 = 0.4
entropy_w = 0.6
entropy_base = ti.field(ti.f32, shape=())
entropy_screen_unit = 0.5 # entropy 1 == ?? screen space


@ti.kernel
def init():
    N[None] = 100

    walls[0] = ti.Vector([xmax, ymin])
    walls[1] = ti.Vector([xmin, ymin])

    walls[2] = ti.Vector([xmin, ymin])
    walls[3] = ti.Vector([xmin, ymax])

    walls[4] = ti.Vector([xmin, ymax])
    walls[5] = ti.Vector([xmax, ymax])

    walls[6] = ti.Vector([xmax, ymin])
    walls[7] = ti.Vector([xmax, ymax])

    lid_q[None] = 0.5
    lid_v[None] = 0.0

    H0[None] = 0.0
    for i in range(N[None]):
        q0[i] = ti.Vector([ti.random(ti.f32) * lid_q[None], ti.random(ti.f32)])
        q1[i] = ti.Vector([ti.random(ti.f32) * (1.0-lid_q[None]) + lid_q[None], 
                           ti.random(ti.f32)])
        # 2 * v^2 * m = E
        v0[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)]) * 0.5
        v1[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32)])
        E0[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5
        E1[None] += m * (v1[i][0] * v1[i][0] + v1[i][1] * v1[i][1]) * 0.5

    lid_e[None] = (E0[None] + E1[None]) / (2*N[None])

    E[None] = E0[None] + E1[None] + lid_e[None] # energy to be conserved (avg_e is)

    for i in range(vis_t*2):
        entropy_vis[i] = ti.Vector([
            entropy_x0 + entropy_w / vis_t*(i//2), entropy_y0])

    entropy_vis_update_counter[None] = 0

@ti.kernel
def update():
    E0[None] = 0
    E1[None] = 0

    # Classical mechanics
    for i in range(N[None]):
        for j in range(N[None]):
            if i != j:
                if ti.math.length(q0[i] - q0[j]) < threshold:
                    tmp = v0[i][0]
                    v0[i][0] = v0[j][0]
                    v0[j][0] = tmp

                    tmp = v0[i][1]
                    v0[i][1] = v0[j][1]
                    v0[j][1] = tmp

                if ti.math.length(q1[i] - q1[j]) < threshold:
                    tmp = v1[i][0]
                    v1[i][0] = v1[j][0]
                    v1[j][0] = tmp

                    tmp = v1[i][1]
                    v1[i][1] = v1[j][1]
                    v1[j][1] = tmp


    #
    for i in range(N[None]):
        q0[i] += dt * v0[i]
        q1[i] += dt * v1[i]

        # Wall boundaries
        if q0[i][0] < 0:
            if v0[i][0] < 0: 
                v0[i][0] = - v0[i][0]
        if q1[i][0] > 1:
            if v1[i][0] > 0: 
                v1[i][0] = - v1[i][0]
        if q0[i][1] < 0:
            if v0[i][1] < 0: 
                v0[i][1] = - v0[i][1]
        if q1[i][1] < 0:
            if v1[i][1] < 0: 
                v1[i][1] = - v1[i][1]
        if q0[i][1] > 1:
            if v0[i][1] > 0: 
                v0[i][1] = - v0[i][1]
        if q1[i][1] > 1:
            if v1[i][1] > 0: 
                v1[i][1] = - v1[i][1]

        # energy exchaning
        if q0[i][0] > lid_q[None]:
            q0[i][0] = lid_q[None]
            if v0[i][0] > 0: 
                v0[i][0] = - v0[i][0]
                # exchange energy
                this_e = (v0[i][0]*v0[i][0] + v0[i][1] * v0[i][1]) * m / 2
                avg_e = (this_e + lid_e[None]) / 2
                k = ti.sqrt( avg_e ) / ti.sqrt( this_e )
                v0[i] = v0[i] * k
                lid_e[None] = avg_e

        if q1[i][0] < lid_q[None]:
            q1[i][0] = lid_q[None]
            if v1[i][0] < 0: 
                v1[i][0] = - v1[i][0]
                # exchange energy
                this_e = (v1[i][0]*v1[i][0] + v1[i][1] * v1[i][1]) * m / 2
                avg_e = (this_e + lid_e[None]) / 2
                k = ti.sqrt( avg_e ) / ti.sqrt( this_e )
                v1[i] = v1[i] * k
                lid_e[None] = avg_e
        
        # keep energy conserve
        E0[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5 
        E1[None] += m * (v1[i][0] * v1[i][0] + v1[i][1] * v1[i][1]) * 0.5 

    current_total = E0[None] + E1[None] + lid_e[None]
    k = ti.sqrt(E[None] / current_total)
    for i in range(N[None]):
        v0[i] = v0[i] * k
        v1[i] = v1[i] * k
    lid_e[None] = lid_e[None] * k * k

    #print(E0[None])

@ti.kernel
def heat_up():
    E0[None] = 0
    E1[None] = 0
    for i in range(N[None]):
        v0[i] = v0[i] * 1.1
        E0[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5
        E1[None] += m * (v1[i][0] * v1[i][0] + v1[i][1] * v1[i][1]) * 0.5
    E[None] = E0[None] + E1[None] + lid_e[None] # energy to be conserved (avg_e is)

@ti.kernel
def cool_down():
    E0[None] = 0
    E1[None] = 0
    for i in range(N[None]):
        v0[i] = v0[i] * 0.9
        E0[None] += m * (v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]) * 0.5
        E1[None] += m * (v1[i][0] * v1[i][0] + v1[i][1] * v1[i][1]) * 0.5
    E[None] = E0[None] + E1[None] + lid_e[None] # energy to be conserved (avg_e is)

@ti.kernel
def update_visual():
    for i in range(maxN):
        q0_vis[i] = ti.Vector([-1, -1])
        q1_vis[i] = ti.Vector([-1, -1])
        
    for i in range(N[None]):
        q0_vis[i][0] = q0[i][0] * (xmax - xmin) + xmin
        q0_vis[i][1] = q0[i][1] * (ymax - ymin) + ymin
        q1_vis[i][0] = q1[i][0] * (xmax - xmin) + xmin
        q1_vis[i][1] = q1[i][1] * (ymax - ymin) + ymin

    lid_vis[0] = ti.Vector([lid_q[None] * (xmax-xmin) + xmin, ymax])
    lid_vis[1] = ti.Vector([lid_q[None] * (xmax-xmin) + xmin, ymin])

    entropy_vis_update_counter[None] -= 1
    if entropy_vis_update_counter[None] <= 0:
        entropy_vis_update_counter[None] = entropy_vis_update_every
        for i in range(0, vis_t-1):
            entropy_vis[i*2][1] = entropy_vis[(i+1)*2][1]

        entropy_vis[(vis_t-1)*2][1] = \
            (entropy[None] - entropy_base[None]) * entropy_screen_unit + entropy_y0

window = ti.ui.Window(
    name="demo", 
    res=(800, 800), pos=(15, 50), fps_limit=100)

init()
ent_base_init = False
while window.running:
    # Compute the entropy
    vs = lid_q[None]/N[None]
    e0 = E0[None]/N[None]
    e1 = E1[None]/N[None]
    s0 = ti.log(vs*ti.sqrt(2*e0*m))
    s1 = ti.log(vs*ti.sqrt(2*e1*m))
    entropy[None] = s0+s1
    if not ent_base_init:
        entropy_base[None] = entropy[None]
        ent_base_init = True

    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.8, 0.8, 0.8))
    hotter = gui.button("Heat Up")
    cooler = gui.button("Cool Down")
    gui.text(f"e0={e0:.3f} s0={s0:.4f} e1={e1:.3f} s1={s1:.4f}, Ent={entropy[None]:.4f}", 
             (0, 0.8, 0, 0.5))

    update()
    update_visual()
    
    if hotter:
        heat_up()
    if cooler:
        cool_down()

    canvas.circles(q0_vis, radius=0.005, color=(0.0, 0.0, 0.5))
    canvas.circles(q1_vis, radius=0.005, color=(0.0, 0.5, 0.0))
    canvas.lines(walls, width=.01)
    canvas.lines(lid_vis, width=.01, color=(0.6, 0, 0))
    canvas.lines(entropy_vis, width=0.01, color=(0.0, 1.0, 0.1, 0.5))
    

    # print(lid)
    # print(lid_v)
    # print(lid_vis)
    
    window.show()
    # time.sleep(1)


import taichi as ti

ti.init(ti.cpu)
PI = ti.math.pi
resolution = 512
particle_num = 2000

dt = ti.field(ti.f32, shape=())
BOUNCE_DAMP = ti.field(ti.f32, shape=())
F = ti.field(ti.f32, shape=())

pos = ti.Vector.field(2, ti.f32, shape=(particle_num,))
v = ti.Vector.field(2, ti.f32, shape=(particle_num,))
f = ti.Vector.field(2, ti.f32, shape=(particle_num,))
density = ti.field(ti.f32, shape=(particle_num,))

LEFT = 0.0
RIGHT = 1.0
TOP = 1.0
BOTTOM = 0.0

H = 0.01                # Smoothing length
MASS = 1.0             # Particle mass
VISCOSITY = 0.01       # Viscosity constant
GAS_CONSTANT = 2000.0  # Gas constant for pressure calculation
REST_DENSITY = 1.0  # Rest density of fluid
GRAV = -9.81           # Gravity constant
BOUNCE_SIZE = 0.5 * H  # Collision bounce threshold


@ti.kernel
def init():
    dt[None] = 0.001
    BOUNCE_DAMP[None] = 0.9
    F[None] = 1.0
    for i in range(particle_num):
        pos[i][0] = ti.random()
        pos[i][1] = ti.random()
        v[i][0] = ti.randn()
        v[i][1] = ti.randn()


window = ti.ui.Window(
    name="SPH", res=(resolution, resolution), pos=(15, 50), fps_limit=100)


@ti.func
def poly6_kernel(r, h):
    """Poly6 kernel for density calculation."""
    result = 0.0
    if 0 <= r < h:
        result = (315 / (64 * PI * h**9)) * (h**2 - r**2)**3
    return result

@ti.func
def spiky_kernel_gradient(r, h):
    """Spiky kernel gradient for pressure force calculation."""
    result = ti.Vector([0.0, 0.0])
    if 0 < r < h:
        factor = -45 / (PI * h**6)
        result = factor * (h - r)**2
    return result

@ti.func
def viscosity_kernel_laplacian(r, h):
    """Viscosity kernel laplacian for viscosity force calculation."""
    result = 0.0
    if 0 <= r < h:
        result = 45 / (PI * h**6) * (h - r)
    return result

@ti.func
def compute_pressure(density):
    """Pressure is computed using an equation of state."""
    return GAS_CONSTANT * (density - REST_DENSITY)


@ti.kernel
def update():
    # Step 1: Compute density for each particle
    for i in range(particle_num):
        density[i] = 0.0
        for j in range(particle_num):
            r = pos[i] - pos[j]
            r_len = ti.math.length(r)
            if r_len < H:
                density[i] += MASS * poly6_kernel(r_len, H)

        print(i, "density", density[i])

    # Step 2: Compute forces for each particle
    for i in range(particle_num):
        f[i] = [0.0, GRAV]  # Reset force, including gravity
        pressure_i = compute_pressure(density[i])
        
        # Pressure and viscosity forces
        bnc = False
        for j in range(particle_num):

            if i != j:
                r = pos[i] - pos[j]
                r_len = ti.math.length(r)
                if r_len < H:
                    bnc = True
                    # Pressure force
                    pressure_j = compute_pressure(density[j])
                    f[i] += -MASS * (pressure_i + pressure_j) / (2 * density[j]) * spiky_kernel_gradient(r_len, H) * (r / r_len)
                    
                    # Viscosity force
                    v_diff = v[j] - v[i]
                    f[i] += VISCOSITY * MASS * (v_diff / density[j]) * viscosity_kernel_laplacian(r_len, H)
        if bnc:
            print(i, f[i])
    for i in range(particle_num):
        pos[i] += dt[None] * v[i]
        v[i] += dt[None] * f[i]

        if pos[i][0] < LEFT: 
            pos[i][0] = LEFT
            v[i][0] = ti.abs(v[i][0]) * BOUNCE_DAMP[None]
        elif pos[i][0] > RIGHT:
            pos[i][0] = RIGHT
            v[i][0] = -ti.abs(v[i][0]) * BOUNCE_DAMP[None]

        if pos[i][1] < BOTTOM: 
            pos[i][1] = BOTTOM
            v[i][1] = ti.abs(v[i][1]) * BOUNCE_DAMP[None]
        elif pos[i][1] > TOP:
            pos[i][1] = TOP
            v[i][1] = -ti.abs(v[i][1]) * BOUNCE_DAMP[None]
            
            


init()
while window.running:
    update()

    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    dt[None] = gui.slider_float("dt", dt[None], minimum=0.0, maximum=0.01)
    F[None] = gui.slider_float("F * 1e6", F[None], minimum=0.001, maximum=100.0)
    canvas.circles(pos, radius=BOUNCE_SIZE, color=(0.0, 0.8, 0.0))
    window.show()
import taichi as ti

ti.init(arch=ti.cpu) 

# Define the resolution of the image
resolution = 512

# Create a Taichi field to store the image
image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution*2, resolution))
WBND = 1.5
vmin = ti.field(ti.f32, shape=())
vmax = ti.field(ti.f32, shape=())
w0 = ti.field(ti.f32, shape=())
w1 = ti.field(ti.f32, shape=())
b = ti.field(ti.f32, shape=())
act = ti.field(ti.i32, shape=())

w_vis = ti.Vector.field(2, ti.f32, shape=(1,))

@ti.kernel
def init():
    vmin[None] = -1
    vmax[None] = +1
    w0[None] = 0.5
    w1[None] = 0.3
    b[None] = 0
    act[None] = 0
    for i, j in image:
        image[i, j] = ti.Vector([.5, .5, .5])

@ti.func
def cmap(s:float) -> ti.Vector:
    c = ti.Vector([0.0, 0.0, 0.0])
    if s < vmin[None]:
        c[2] = 0.5
    elif s > vmax[None]:
        c[0] = 1.0
    else:
        s1 = (s-vmin[None]) / (vmax[None] - vmin[None])
        c[0] = s1
        c[2] = 1-s1
    return c
    

@ti.kernel
def compute_image():
    for i, j in image:
        if i >= resolution:
            x0 = (float(i-resolution) / resolution) - 0.5
            x1 = (float(j) / resolution) - 0.5
            s = w0[None] * x0 + w1[None] * x1 + b[None]
            if act[None]:
                if s > 0:
                    s = 1.0
                else:
                    s = -1.0
            image[i, j] = cmap(s)
        else:
            image[i, j] = ti.Vector([0.3, 0.3, 0.3])

@ti.kernel
def compute_w_vis():
    w_vis[0][0] = (w0[None] - (-WBND)) / (2*WBND) * 0.5
    w_vis[0][1] = (w1[None] - (-WBND)) / (2*WBND)
    return

    

# Create a GUI window
window = ti.ui.Window(
    name="Model Out", res=(resolution*2, resolution), pos=(15, 50), fps_limit=100)

init()
while window.running:
    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    w0[None] = gui.slider_float("W0", w0[None], minimum=-WBND, maximum=WBND)
    w1[None] = gui.slider_float("W1", w1[None], minimum=-WBND, maximum=WBND)
    b[None] = gui.slider_float("b", b[None], minimum=-WBND, maximum=WBND)
    act_pressed = gui.button("Toggle ACT")
    if act_pressed:
        act[None] = 1-act[None]

    compute_image()
    compute_w_vis()
    img_np = image.to_numpy()
    canvas.set_image(img_np)
    canvas.circles(w_vis, radius=0.01, color=(1, 1, 0))
    
    # Display the image in the GUI
    window.show()
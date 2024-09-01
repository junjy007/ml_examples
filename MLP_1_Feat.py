import taichi as ti

ti.init(arch=ti.cpu) 

# Define the resolution of the image
resolution = 512

# Create a Taichi field to store the image
image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution, resolution))
WBND = 1.5
XBND = 1.0
vmin = ti.field(ti.f32, shape=())
vmax = ti.field(ti.f32, shape=())
w0 = ti.field(ti.f32, shape=())
w1 = ti.field(ti.f32, shape=())
w2 = ti.field(ti.f32, shape=())
w3 = ti.field(ti.f32, shape=())
w4 = ti.field(ti.f32, shape=())
w5 = ti.field(ti.f32, shape=())
w6 = ti.field(ti.f32, shape=())
b = ti.field(ti.f32, shape=())
act = ti.field(ti.i32, shape=())

w_vis = ti.Vector.field(2, ti.f32, shape=(1,))

@ti.kernel
def init():
    vmin[None] = -1
    vmax[None] = +1
    w0[None] = 0.
    w1[None] = 0.
    w2[None] = 0.
    w3[None] = 0.
    w4[None] = 0.
    w5[None] = 0.
    w6[None] = 0.
    b[None] = 0
    act[None] = 0
    for i, j in image:
        image[i, j] = ti.Vector([.5, .5, .5])

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
        c[2] = 1-s1
    return c
    

@ti.kernel
def compute_image():
    for i, j in image:
        x0 = (float(i) / resolution) * 2.0 * XBND + (-XBND)
        x1 = (float(j) / resolution) * 2.0 * XBND + (-XBND)
        s = w0[None] * x0 + w1[None] * x1 \
            + w2[None] * x0 * x1 \
            + w3[None] * x0 * x0 \
            + w4[None] * x1 * x1 \
            + w5[None] * ti.math.cos(x0*3.14*2) \
            + w6[None] * ti.math.cos(x1*3.14*2) \
            + b[None]
        if act[None]:
            if s > 0:
                s = 1.0
            else:
                s = -1.0
        image[i, j] = cmap(s)


# Create a GUI window
window = ti.ui.Window(
    name="Model Out", res=(resolution, resolution), pos=(15, 50), fps_limit=100)

init()
while window.running:
    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    w0[None] = gui.slider_float("W0", w0[None], minimum=-WBND, maximum=WBND)
    w1[None] = gui.slider_float("W1", w1[None], minimum=-WBND, maximum=WBND)
    w2[None] = gui.slider_float("W2", w2[None], minimum=-WBND, maximum=WBND)
    w3[None] = gui.slider_float("W3", w3[None], minimum=-WBND, maximum=WBND)
    w4[None] = gui.slider_float("W4", w4[None], minimum=-WBND, maximum=WBND)
    w5[None] = gui.slider_float("W5", w5[None], minimum=-WBND, maximum=WBND)
    w6[None] = gui.slider_float("W6", w6[None], minimum=-WBND, maximum=WBND)
    b[None] = gui.slider_float("b", b[None], minimum=-WBND, maximum=WBND)
    act_pressed = gui.button("Toggle ACT")
    if act_pressed:
        act[None] = 1-act[None]
    reset_pressed = gui.button("Reset")
    if reset_pressed:
        init()

    compute_image()
    img_np = image.to_numpy()
    canvas.set_image(img_np)
    
    # Display the image in the GUI
    window.show()
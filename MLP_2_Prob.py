import taichi as ti

ti.init(arch=ti.cpu) 

# Define the resolution of the image
resolution = 512

# Create a Taichi field to store the image
image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution*2, resolution))
WBND = 1.5
XBND = 1.
WSELE_TH = 0.1
WN = 1000
XN = 1000
MINLOGP = -6
vmin = ti.field(ti.f32, shape=())
vmax = ti.field(ti.f32, shape=())
W = ti.Vector.field(2, ti.f32, shape=(WN,))
logPW = ti.field(ti.f32, shape=(WN,))
w_MAP_i = ti.field(ti.i32, shape=())

b = ti.field(ti.f32, shape=())
act = ti.field(ti.i32, shape=())
# ... for selecting W
wi_sele = ti.field(ti.i32, shape=())
attemp_w = ti.Vector.field(2, ti.f32, shape=(1,))
# ... for managing X
X = ti.Vector.field(2, ti.f32, shape=(XN,))
Y = ti.field(ti.f32, shape=(XN,))
N = ti.field(ti.i32, shape=())
attemp_x = ti.Vector.field(2, ti.f32, shape=(1,))
X_vis = ti.Vector.field(2, ti.f32, shape=(XN,))
X_vis_clr = ti.Vector.field(3, ti.f32, shape=(XN,))

W_vis = ti.Vector.field(2, ti.f32, shape=(WN,))
W_vis_clr = ti.Vector.field(3, ti.f32, shape=(WN,))
Wsele_vis = ti.Vector.field(2, ti.f32, shape=(1,))

# flags
flag_select_w = ti.field(ti.i32, shape=())
flag_select_x = ti.field(ti.i32, shape=()) # 0 invalid, 1 add 2 remove

@ti.kernel
def init():
    vmin[None] = -1.0
    vmax[None] = +1.0
    b[None] = 0
    act[None] = 0
    wi_sele[None] = -1
    flag_select_w[None] = 0
    flag_select_x[None] = 0
    N[None] = 0
    for i, j in image:
        image[i, j] = ti.Vector([.5, .5, .5])
    for i in range(WN):
        W[i][0] = ti.randn() * 0.5
        W[i][1] = ti.randn() * 0.5
        logPW[i] = ti.log(1/WN)

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
def cmapw(s:float) -> ti.Vector: # s: log Pw
    c = ti.Vector([0.0, 0.0, 0.0])
    if s > MINLOGP:
        s1 = ti.exp(s)
        if s1 > 1:
            s1 = 1.0
        c[0] = s1
        c[1] = s1
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
        if i >= resolution:
            x0 = (float(i-resolution) / resolution) * (2*XBND) - (XBND)
            x1 = (float(j) / resolution) * (2*XBND) - (XBND)
            s = 0.0
            if wi_sele[None] >= 0:
                wi = wi_sele[None]
                w0 = W[wi_sele[None]][0]
                w1 = W[wi_sele[None]][1]
                s = w0 * x0 + w1 * x1 + b[None]
                if act[None]:
                    s = actf(s)
            else:
                for k in range(WN):
                    w0 = W[k][0]
                    w1 = W[k][1]
                    sk = w0 * x0 + w1 * x1 + b[None]
                    if act[None]:
                        sk = actf(sk)
                    if logPW[k] > MINLOGP:
                        s += sk * ti.exp(logPW[k])
            image[i, j] = cmap(s)
        else:
            image[i, j] = ti.Vector([0.3, 0.3, 0.3])

@ti.kernel
def compute_w_vis():
    max_log_pw = -99999.0
    for k in range(WN):
        if max_log_pw < logPW[k]:
            max_log_pw = logPW[k]

    for k in range(WN):
        W_vis[k][0] = (W[k][0] - (-WBND)) / (2*WBND) * 0.5
        W_vis[k][1] = (W[k][1] - (-WBND)) / (2*WBND)
        W_vis_clr[k] = cmapw(logPW[k] - max_log_pw)
    if wi_sele[None] >= 0:
        Wsele_vis[0][0] = (W[wi_sele[None]][0] - (-WBND)) / (2*WBND) * 0.5
        Wsele_vis[0][1] = (W[wi_sele[None]][1] - (-WBND)) / (2*WBND)
        #print("sele:", W[wi_sele[None]], "scr", Wsele_vis)
    
@ti.kernel
def select_w():
    d = 999.0
    j = -1
    for i in range(WN):
        di = ti.math.length(W[i] - attemp_w[0])
        if di < d:
            d = di
            j = i
            # print(di, j)
    if d < WSELE_TH:
        wi_sele[None] = j

@ti.kernel
def compute_x_vis():
    X_vis.fill(-1)
    for i in range(N[None]):
        X_vis[i][0] = (X[i][0] - (-XBND)) / (2*XBND) * 0.5+ 0.5
        X_vis[i][1] = (X[i][1] - (-XBND)) / (2*XBND)
        if Y[i] >= 0:
            X_vis_clr[i] = ti.Vector([0.5, 0.1, 0])
        else:
            X_vis_clr[i] = ti.Vector([0.0, 0.1, 0.5])

@ti.kernel
def compute_w_scores():
    wlike_most = -9999.0
    wz = 0.0
    w_MAP_i[None] = -1
    for i in range(WN):
        wlike = 0.0
        for n in range(N[None]):
            logit = W[i][0] * X[n][0] + W[i][1] * X[n][1] + b[None]
            if Y[n] > 0:
                wlike += ti.log(1/(1+ti.exp(-logit)))
            else:
                wlike -= ti.log(1/(1+ti.exp(-logit)))

        logPW[i] = wlike
        wz += ti.exp(wlike)
        if wlike > wlike_most:
            wlike_most = wlike
            w_MAP_i[None] = i

    wz = ti.log(wz)
    print(wz)
    for i in range(WN):
        logPW[i] -= wz

    

# Create a GUI window
window = ti.ui.Window(
    name="Model Out", res=(resolution*2, resolution), pos=(15, 50), fps_limit=100)

init()
while window.running:
    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    b[None] = gui.slider_float("b", b[None], minimum=-WBND, maximum=WBND)

    act_pressed = gui.button("Toggle ACT")
    if act_pressed:
        act[None] = 1-act[None]

    # toggle selecting a W-hypo to show
    sele_w_pressed = gui.button("Clear/Stop W Select" if flag_select_w[None] else "W Select")
    if sele_w_pressed:
        flag_select_w[None] = 1 - flag_select_w[None]

    # toggle adding / removing data points
    msg = ""
    if flag_select_x[None] == 0:
        msg = "Add X"
    if flag_select_x[None] == 1:
        msg = "Lock X"
    sele_x_pressed = gui.button(msg)
    if sele_x_pressed:
        flag_select_x[None] += 1
        if flag_select_x[None] == 2:
            flag_select_x[None] = 0


    mouse = window.get_cursor_pos()
    w_evt = False
    x_evt = False
    if mouse[0] < 0.5:
        attemp_w[0][0] = (mouse[0] / 0.5) * (2*WBND) + (-WBND)
        attemp_w[0][1] = (mouse[1]) * (2*WBND) + (-WBND)
        w_evt = (flag_select_w[None] == 1)
    else:
        attemp_x[0][0] = (mouse[0] - 0.5) / 0.5 * (2 * XBND) + (-XBND)
        attemp_x[0][1] = (mouse[1]) * (2 * XBND) + (-XBND)
        x_evt = (flag_select_x[None] == 1)

    if window.get_event(ti.ui.RELEASE):
        if window.event.key == ti.ui.LMB and w_evt:
            select_w()
        
        if window.event.key == ti.ui.LMB and x_evt:
            X[N[None]] = attemp_x[0]
            Y[N[None]] = 1.0
            N[None] += 1
        if window.event.key == ti.ui.RMB and x_evt:
            X[N[None]] = attemp_x[0]
            Y[N[None]] = -1.0
            N[None] += 1

    if flag_select_w[None] == 0:
        wi_sele[None] = -1
    

    compute_w_scores()
    compute_image()
    compute_w_vis()
    compute_x_vis()
    img_np = image.to_numpy()
    canvas.set_image(img_np)
    canvas.circles(W_vis, radius=0.003, per_vertex_color=W_vis_clr)
    canvas.circles(X_vis, radius=0.01, per_vertex_color=X_vis_clr)
    if wi_sele[None] >= 0:
        canvas.circles(Wsele_vis, radius=0.005, color=(1.0, 1.0, 0))
    

    
    # Display the image in the GUI
    window.show()
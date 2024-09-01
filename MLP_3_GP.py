import taichi as ti
import numpy as np

ti.init(arch=ti.cpu) 

# Define the resolution of the image
resolution = 512

# Create a Taichi field to store the image
image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution, resolution))
WBND = 1.5
XBND = 1.0
MINLOGP = -8
WN = 1000
HD = 6
XN = 1000
vmin = ti.field(ti.f32, shape=())
vmax = ti.field(ti.f32, shape=())
logPW = ti.field(ti.f32, shape=(WN,))
w_MAP_i = ti.field(ti.i32, shape=())
W = ti.Vector.field(HD, ti.f32, shape=(WN,))
RegW = ti.Vector.field(HD, ti.f32, shape=(1,))
act = ti.field(ti.i32, shape=())
# ... for managing X
X = ti.Vector.field(2, ti.f32, shape=(XN,))
Y = ti.field(ti.f32, shape=(XN,))
N = ti.field(ti.i32, shape=())
attemp_x = ti.Vector.field(2, ti.f32, shape=(1,))
X_vis = ti.Vector.field(2, ti.f32, shape=(XN,))
X_vis_clr = ti.Vector.field(3, ti.f32, shape=(XN,))
# Design Gram and Kernel Matrices
R = ti.field(ti.f32, shape=(XN, HD))
LR = 0.002
LAMBDA = 0.001
sigma2 = ti.field(ti.f32, shape=())
K = ti.field(ti.f32, shape=(XN, XN))
Kinv = ti.field(ti.f32, shape=(XN, XN))
KinvY = ti.field(ti.f32, shape=(XN,))
kvec = ti.field(ti.f32, shape=(XN,))

w_vis = ti.Vector.field(2, ti.f32, shape=(1,))

flag_select_x = ti.field(ti.i32, shape=())
# which data model
# 1. Monte Carlo MAP - W
# 2. Monte Carlo Bayes - W
# 3. Solve W
# 4. Kernel Regress y (as -1, +1 )
# 5. Gaussian Process (sample a posterior)
model_id = ti.field(ti.i32, shape=())


@ti.kernel
def init():
    vmin[None] = -1
    vmax[None] = +1
    act[None] = 0
    N[None] = 0
    model_id[None] = 0
    w_MAP_i[None] = 0
    sigma2[None] = 0.1
    for i, j in image:
        image[i, j] = ti.Vector([.5, .5, .5])
    for i in range(WN):
        for j in range(HD):
            W[i][j] = ti.randn() * .5
        logPW[i] = ti.log(1/WN)
    for i, j in R:
        R[i, j] = 0.0
    for i in range(HD):
        RegW[0][i] = 0.0
    for i in range(XN):
        kvec[i] = 0
        KinvY[i] = 0
        for j in range(XN):
            K[i, j] = 0
            Kinv[i, j] = 0

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
    
@ti.func
def actf(s:ti.f32) -> ti.f32:
    if s > 0:
        s = 1.0
    else:
        s = -1.0
    return s

@ti.func
def logit_W_X(w:ti.types.vector(HD, ti.f32), x:ti.types.vector(2, ti.f32)) -> ti.f32:
    sp = w[0] * x[0] + w[1] * x[1] \
        + w[2] * x[0] * x[1] \
        + w[3] * x[0] * x[0] \
        + w[4] * x[1] * x[1] \
        + w[5] 
    return sp

@ti.func
def pred_X(x:ti.types.vector(2, ti.f32)) -> ti.f32:
    s = ti.f32(0.0)
    if model_id[None] == 0 and w_MAP_i[None] >= 0:
        wi = w_MAP_i[None]
        s = logit_W_X(W[wi], x)
        if act[None]:
            s = actf(s) 

    elif model_id[None] == 1:
        for p in range(WN):
            sp = logit_W_X(W[p], x)
            if act[None]:
                sp = actf(sp) 
            if logPW[p] > MINLOGP:
                s += sp * ti.exp(logPW[p])

    elif model_id[None] == 2: # solved W
        s = logit_W_X(RegW[0], x)
        if act[None]:
            s = actf(s) 

    elif model_id[None] == 3 and N[None] > 0: # kernel regression
        weights = 0.0
        weighted_sum = 0.0
        for i in range(N[None]):
            weight = kf(x, X[i])
            weights += weight
            weighted_sum += weight * Y[i]
        if weights == 0:
            s = 0.0
        else:
            s = weighted_sum / weights
        if act[None]:
            s = actf(s) 

    elif model_id[None] == 4 and N[None] > 0:
        for i in range(N[None]):
            kvec[i] = kf(x, X[i])
        kself = kf(x, x)
        for i in range(N[None]):
            s = s + kvec[i] * KinvY[i]
        kk = 0.0
        for i in range(N[None]):
            for j in range(N[None]):
                kk += Kinv[i,j] * kvec[i] * kvec[j]
        vr = kself - kk
        if vr < 0:
            vr = 0
        s = ti.randn() * ti.math.sqrt(vr) + s
        if act[None] > 0:
            s = actf(s)

        print(s)
    return s
    
@ti.kernel
def compute_image_ker():
    for i, j in image:
        x0 = (float(i) / resolution) * 2.0 * XBND + (-XBND)
        x1 = (float(j) / resolution) * 2.0 * XBND + (-XBND)
        s = pred_X(ti.Vector([x0, x1], ti.f32))
        image[i, j] = cmap(s)

def compute_image():
    if model_id[None] != 99:
        compute_image_ker()
    else:
        xx, yy = np.meshgrid(np.arange(resolution), np.arange(resolution))
        xxx = np.stack([xx.flatten(), yy.flatten()]).T
        print(xxx.shape)
        xxx /= resolution
        xxx *= (2 * XBND)
        xxx -= XBND

@ti.kernel
def compute_x_vis():
    X_vis.fill(-1)
    for i in range(N[None]):
        X_vis[i][0] = (X[i][0] - (-XBND)) / (2*XBND) 
        X_vis[i][1] = (X[i][1] - (-XBND)) / (2*XBND)
        if Y[i] >= 0:
            X_vis_clr[i] = ti.Vector([0.5, 0.1, 0])
        else:
            X_vis_clr[i] = ti.Vector([0.0, 0.1, 0.5])

# Data model
@ti.kernel
def compute_w_scores():
    wlike_most = -9999.0
    wz = 0.0
    w_MAP_i[None] = -1
    for i in range(WN):
        wlike = 0.0
        for n in range(N[None]):
            # logit = W[i][0] * X[n][0] + W[i][1] * X[n][1] \
            #     + W[i][2] * X[n][0] * X[n][1] \
            #     + W[i][3] * X[n][0] * X[n][0] \
            #     + W[i][4] * X[n][1] * X[n][1] \
            #     + W[i][5] 
            logit = logit_W_X(W[i], X[n])
            if Y[n] > 0:
                wlike += ti.log(1/(1+ti.exp(-logit)))
            else:
                wlike -= ti.log(1/(1+ti.exp(-logit)))

        logPW[i] = wlike
        wz += ti.exp(wlike)
        if wlike > wlike_most:
            wlike_most = wlike
            w_MAP_i[None] = i
            print(wlike, i)

    wz = ti.log(wz)
    print(wz)
    for i in range(WN):
        logPW[i] -= wz

@ti.kernel
def update_R():
    for i in range(N[None]):
        x0 = X[i][0]
        x1 = X[i][1]
        R[i, 0] = x0
        R[i, 1] = x1
        R[i, 2] = x0*x1
        R[i, 3] = x0*x0
        R[i, 4] = x1*x1
        R[i, 5] = 1.0

@ti.kernel
def solve_W(num_iterations: int):
    if N[None] > 0:
        n = float(N[None])
        for _ in range(num_iterations):
            # Gradient descent update
            for j in range(HD):
                grad = 0.0
                for i in range(n):
                    predicted = 0.0
                    for k in range(HD):
                        predicted += RegW[0][k] * R[i, k]
                    grad += (predicted - Y[i]) * R[i, j]
                RegW[0][j] -= LR * (grad / n - LAMBDA * RegW[0][j]) # Update rule with learning rate and average over samples
        print(RegW[0])

@ti.func
def kf(x:ti.types.vector(2, ti.f32), y:ti.types.vector(2, ti.f32)):
    d = ti.math.length(x - y)
    k = ti.exp( -0.5 * (d*d) / sigma2[None])
    return k

@ti.kernel
def compute_kernel_matrix_k():
    n = N[None]
    if n > 0:
        for i, j in ti.ndrange(N[None], N[None]):
            K[i, j] = kf(X[i], X[j])

def compute_kernel_matrix():
    n = N[None]
    if N[None] > 0:
        compute_kernel_matrix_k()
        Y_np = Y.to_numpy()
        Ki_ = np.linalg.inv(K.to_numpy()[:n, :n] + sigma2[None] * np.eye(n))
        Kifull = np.zeros((XN, XN), dtype=np.float32)
        Kifull[:n, :n] = Ki_
        Ki_Y = np.dot(Kifull, Y_np)

        Kinv.from_numpy(Kifull)
        KinvY.from_numpy(Ki_Y)
        print("kernel inv Y", Ki_Y[:n])
        print("kernel inv Y", Ki_)



# Create a GUI window
window = ti.ui.Window(
    name="Model Out", res=(resolution, resolution), pos=(15, 50), fps_limit=100)

init()
iter_t = 0
x_evt = True
while window.running:
    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    act_pressed = gui.button("Toggle ACT")
    if act_pressed:
        act[None] = 1-act[None]
        x_evt = True
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
    # data models
    msgs = ["1. MC-MAP", "2. MC-Bayes", "3. Solve W", "4. Kernel Regress", "5. GP"]
    mi = model_id[None]
    next_msg = msgs[ (mi+1) % 5]
    gui.text(f"Mode: {msgs[mi]}")
    change_mode_pressed = gui.button(f"Change model to {next_msg}")
    if change_mode_pressed:
        model_id[None] = (model_id[None] + 1) % 5
        x_evt = True

    reset_pressed = gui.button("Reset")
    if reset_pressed:
        init()
        x_evt = True


    mouse = window.get_cursor_pos()
    attemp_x[0][0] = (mouse[0]) * (2 * XBND) + (-XBND)
    attemp_x[0][1] = (mouse[1]) * (2 * XBND) + (-XBND)

    if window.get_event(ti.ui.RELEASE) and flag_select_x[None]:
        if window.event.key == ti.ui.LMB:
            X[N[None]] = attemp_x[0]
            Y[N[None]] = 1.0
            N[None] += 1
        if window.event.key == ti.ui.RMB:
            X[N[None]] = attemp_x[0]
            Y[N[None]] = -1.0
            N[None] += 1
        x_evt = True

    if x_evt:
        update_R()
        compute_kernel_matrix()

        compute_w_scores()
        solve_W(1000)
        compute_image()
        x_evt = False
        if model_id[None] == 4:
            x_evt = True
    img_np = image.to_numpy()
    canvas.set_image(img_np)
    compute_x_vis()
    canvas.circles(X_vis, radius=0.01, per_vertex_color=X_vis_clr)
    
    iter_t += 1
    # Display the image in the GUI
    window.show()
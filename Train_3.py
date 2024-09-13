import taichi as ti
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Model
hdim = 16
hdim2 = 4
feature_layer1 = nn.Linear(2, hdim)
feature_layer2 = nn.Linear(hdim, 2)
w_layer = nn.Linear(2, hdim2, bias=False)
out_layer = nn.Linear(hdim2, 1, bias=False)
# act = torch.tanh
act = torch.relu
def feat_model(x):
    h1 = feature_layer1(x)
    h1 = act(h1)
    h2 = feature_layer2(h1)
    
    return h2

def model(x):
    h2 = feat_model(x)
    h2 = act(h2)
    h2 = w_layer(h2)
    h2 = act(h2)
    return out_layer(h2)

def multiple_models(x, W):
    # compute output using multiple versions of W
    # args
    # - x: XN x 2
    # - W: WN x 2
    # Note See comments
    # First compute features to x -> h2 [XN x 2]
    XN = x.shape[0]
    WN = W.shape[0]
    h2 = feat_model(x)
    h2 = act(h2)
    # Compute intermediate features of multiple version of models
    h3 = torch.einsum('nj,mj->nm', h2, W) # #.samples x #.models
    h3 = h3.reshape(-1, 1)
    h2 = h2.unsqueeze(dim=1)
    h2 = h2.expand(XN, WN, 2).reshape(-1, 2)
    h3_all = w_layer(h2)
    h3_all[:, 0] = h3[:, 0]
    h3_all = act(h3_all)
    return out_layer(h3_all).reshape(XN, WN)

def init_model(seed):
    torch.manual_seed(seed)
    nn.init.normal_(feature_layer1.weight, mean=0.0, std=math.sqrt(1./2.0))
    # nn.init.normal_(feature_layer1.bias, mean=0.0, std=math.sqrt(1.)/2)
    nn.init.normal_(feature_layer2.weight, mean=0.0, std=math.sqrt(1./hdim))
    # nn.init.normal_(feature_layer2.bias, mean=0.0, std=math.sqrt(1.)/2)
    nn.init.normal_(w_layer.weight, mean=0.0, std=math.sqrt(1./hdim2))
    # nn.init.normal_(w_layer.bias, mean=0.0, std=math.sqrt(1.)/2)
    nn.init.normal_(out_layer.weight, mean=0.0, std=math.sqrt(1./2.0)*3)

def set_out_w(w):
    # setup the first weight
    w_layer.weight.data[0, 0] = w[0]
    w_layer.weight.data[0, 1] = w[1]

def get_out_w_grad():
    g = [0, 0]
    if out_layer.weight.grad is not None:
        g[0] = out_layer.weight.grad[0]
        g[1] = out_layer.weight.grad[1]

# Train Data
def generate_training_samples(seed, 
    num_samples=1000, r0=0.2, r1=0.5, r2=0.7):
    rng = np.random.RandomState(seed)
    
    # Number of samples for each class
    num_samples_class1 = num_samples // 2
    num_samples_class2 = num_samples // 2
    
    # Class 1: points on a ring (with radii between r1 and r2)
    angles_class1 = rng.uniform(0, 2 * np.pi, num_samples_class1)
    radii_class1 = rng.uniform(r1, r2, num_samples_class1)
    x_class1 = radii_class1 * np.cos(angles_class1)
    y_class1 = radii_class1 * np.sin(angles_class1)
    
    # Class 2: points within a circle of radius r0
    angles_class2 = rng.uniform(0, 2 * np.pi, num_samples_class2)
    radii_class2 = r0 * np.sqrt(rng.uniform(0, 1, num_samples_class2))  # sqrt to ensure uniform distribution within a circle
    x_class2 = radii_class2 * np.cos(angles_class2)
    y_class2 = radii_class2 * np.sin(angles_class2)
    
    # Combine class 1 and class 2 data
    X_class1 = np.vstack((x_class1, y_class1)).T
    X_class2 = np.vstack((x_class2, y_class2)).T
    
    # Labels for class 1 and class 2
    y_class1 = np.ones(num_samples_class1)
    y_class2 = np.zeros(num_samples_class2)
    
    # Combine the datasets
    X_trn = np.vstack((X_class1, X_class2))
    y_trn = np.hstack((y_class1, y_class2))
    
    return X_trn + np.array([1.5, 0.5]), y_trn

def generate_training_samples2(seed, 
    num_samples=100, num_clusters=3, xboundary=5.0, std_dev=0.5):

    rng = np.random.RandomState(seed)
    
    # Number of samples per class (split equally for simplicity)
    num_samples_per_class = num_samples // 2
    
    # Define centers of clusters randomly within the +/- xboundary
    centers_class1 = rng.uniform(-xboundary, xboundary, (num_clusters, 2))
    centers_class2 = rng.uniform(-xboundary, xboundary, (num_clusters, 2))
    
    # Generate Gaussian clusters for class 1
    X_class1 = []
    for center in centers_class1:
        samples = rng.normal(loc=center, scale=std_dev, size=(num_samples_per_class // num_clusters, 2))
        X_class1.append(samples)
    X_class1 = np.vstack(X_class1)
    
    # Generate Gaussian clusters for class 2
    X_class2 = []
    for center in centers_class2:
        samples = rng.normal(loc=center, scale=std_dev, size=(num_samples_per_class // num_clusters, 2))
        X_class2.append(samples)
    X_class2 = np.vstack(X_class2)
    
    # Labels for class 1 and class 2
    y_class1 = np.ones(X_class1.shape[0])
    y_class2 = np.zeros(X_class2.shape[0])
    
    # Combine the datasets
    X_trn = np.vstack((X_class1, X_class2))
    y_trn = np.hstack((y_class1, y_class2))
    
    return X_trn, y_trn

# Init Visualisation
ti.init(arch=ti.cpu)

resolution = 512

image = ti.Vector.field(3, dtype=ti.f32, shape=(resolution*2, resolution))
ximage = ti.field(ti.f32, shape=(resolution, resolution))
wimage = ti.field(ti.f32, shape=(resolution, resolution))
WBND = ti.field(ti.f32, shape=())
XBND = ti.field(ti.f32, shape=())
W_curr = ti.Vector.field(2, ti.f32, shape=())
WVIS_NUM = 1000
wvis_n = ti.field(ti.i32, shape=())
W_curr_vis = ti.Vector.field(2, ti.f32, shape=(WVIS_NUM,))
ymin = ti.field(ti.f32, shape=())
ymax = ti.field(ti.f32, shape=())
lmin = ti.field(ti.f32, shape=())
lmax = ti.field(ti.f32, shape=())
wscale = ti.field(ti.f32, shape=())
L2w = ti.field(ti.f32, shape=())
W_grad = ti.Vector.field(2, ti.f32, shape=())

# optimisation
stepsize = ti.field(ti.f32, shape=())
batchsize = 4
batchmode = False

# visualisation for training samples
X_train_num = 96 * 6
X_train = ti.Vector.field(2, ti.f32, shape=(X_train_num,))
X_train_vis = ti.Vector.field(2, ti.f32, shape=(X_train_num,))
X_batch_vis = ti.Vector.field(2, ti.f32, shape=(batchsize,))
X_train_vis_clr = ti.Vector.field(3, ti.f32, shape=(X_train_num))

# == X-Space computation and vis ==
@ti.func
def xcmap(s:float) -> ti.Vector:
    c = ti.Vector([0.0, 0.0, 0.0])
    if s < ymin[None]:
        c[2] = 1.0
    elif s > ymax[None]:
        c[0] = 1.0
    else:
        s1 = (s-ymin[None]) / (ymax[None] - ymin[None])
        c[0] = s1
        c[2] = 1.0 - s1
    return c

@ti.kernel
def fill_x_image():
    for xi in range(resolution):
        for yi in range(resolution):
            image[resolution + xi, yi] = xcmap(ximage[xi, yi])

def compute_X_image():
    xb = XBND[None]
    xvs = torch.linspace(-xb, xb, resolution)
    xx0, xx1 = torch.meshgrid(xvs, xvs)
    xx = torch.stack([xx0.flatten(), xx1.flatten()]).T
    out = model(xx)
    out = out.detach().numpy()\
        .reshape((resolution, resolution)).astype(np.float32)
    ximage.from_numpy(out)
    fill_x_image()

# @ti.kernel
def compute_X_train_vis(X, Y):
    X_train_vis.fill(-1)
    xb = XBND[None]
    for i in range(X_train_num):
        X_train_vis[i][0] = (X[i][0] + xb) / (2*xb) * 0.5+ 0.5
        X_train_vis[i][1] = (X[i][1] + xb) / (2*xb)
        if Y[i] > 0:
            X_train_vis_clr[i] = ti.Vector([0.5, 0.1, 0])
        else:
            X_train_vis_clr[i] = ti.Vector([0.0, 0.1, 0.5])

def compute_X_batch_vis(X_batch):
    X_batch_vis.fill(-1)
    xb = XBND[None]
    for i in range(batchsize):
        X_batch_vis[i][0] = (X_batch[i][0] + xb) / (2*xb) * 0.5+ 0.5
        X_batch_vis[i][1] = (X_batch[i][1] + xb) / (2*xb)


# == W-Space computation and vis ==
@ti.func
def wcmap(s:float) -> ti.Vector:
    c = ti.Vector([0.0, 0.0, 0.0])

    s0 = lmin[None]
    s1 = lmin[None] +(lmax[None] - lmin[None]) * wscale[None]

    c0 = [0.0, 0.5, 0.0]
    c1 = [1.0, 1.0, 0.0]
    if s < s0:
        c[0] = c0[0]
        c[1] = c0[1]
        c[2] = c0[2]
    elif s > s1:
        c[0] = c1[0]
        c[1] = c1[1]
        c[2] = c1[2]
    
    else:
        s1 = (s-s0) / (s1-s0)
        c[0] = s1 * c1[0] + (1-s1) * c0[0]
        c[1] = s1 * c1[1] + (1-s1) * c0[1]
        c[2] = s1 * c1[2] + (1-s1) * c0[2]
    return c

@ti.kernel
def fill_w_image():
    for xi in range(resolution):
        for yi in range(resolution):
            image[xi, yi] = wcmap(wimage[xi, yi])


def compute_W_image(X_trn, Y_trn):
    wb = WBND[None]
    wvs = torch.linspace(-wb, wb, resolution)
    ww0, ww1 = torch.meshgrid(wvs, wvs)
    ww = torch.stack([ww0.flatten(), ww1.flatten()]).T
    X = torch.Tensor(X_trn)
    Y = torch.LongTensor(Y_trn)

    # computer features:
    # h2 = feat_model(X)
    # out = torch.einsum('nj,kj->nk', h2, ww) # #.train_samples x #.models
    out = multiple_models(X, ww)

    # Compute cross-entropy loss manually for binary classification 
    # (assuming 2 classes: 0 and 1)
    # Apply sigmoid to each model's logits and compute binary cross-entropy
    Y = Y.float()  # Make sure Y is float for binary classification loss
    # print(out.shape)
    L = F.binary_cross_entropy_with_logits(out, Y.unsqueeze(1).expand_as(out), reduction='none')
    # Now L is #.samples x #. models
    L = L.detach().mean(dim=0).reshape(resolution, resolution).numpy()
    

    W_mag = (ww * ww).sum(dim=1)
    W_mag = W_mag.reshape(resolution, resolution).numpy()
    L1 = L + W_mag * L2w[None] * 1e-5
    
    wimage.from_numpy(L1)

    lmin[None] = L1.min() 
    lmax[None] = L1.max()
    # fill_w_image()
    # print(L2w[None], lmin[None], lmax[None])
    # print(W_mag[::100, ::100])
    # print(L[::100, ::100])
    return L
    
@ti.kernel
def update_W_vis(do_record:bool):
    wi = wvis_n[None]
    W_curr_vis[wi][0] = (W_curr[None][0] + WBND[None]) / (2*WBND[None]) * 0.5
    W_curr_vis[wi][1] = (W_curr[None][1] + WBND[None]) / (2*WBND[None])
    if do_record:
        wvis_n[None] += 1
        if wvis_n[None] >= WVIS_NUM:
            wvis_n[None] = 0
    

def compute_W_grad(X_trn, Y_trn):
    for m in [feature_layer1, feature_layer2, w_layer, out_layer]:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()

    set_out_w(W_curr[None])
    X = torch.Tensor(X_trn)
    Y = torch.LongTensor(Y_trn).unsqueeze(dim=1).to(torch.float32)
    out = model(X)
    L = F.binary_cross_entropy_with_logits(out, Y, reduction='mean')
    L += (w_layer.weight[0, 0] ** 2 + w_layer.weight[0, 1] ** 2) * L2w[None] * 1e-5
    L.backward()
    print("L", L, w_layer.weight.grad[0], X.shape, Y.shape)
    W_grad[None][0] = w_layer.weight.grad[0, 0]
    W_grad[None][1] = w_layer.weight.grad[0, 1]

@ti.kernel
def update_W_curr():
    s_ = ti.math.exp(stepsize[None] * ti.math.log(10.0) )
    W_curr[None][0] = W_curr[None][0] - W_grad[None][0] * s_
    W_curr[None][1] = W_curr[None][1] - W_grad[None][1] * s_

@ti.kernel
def init():
    WBND[None] = 2.0
    XBND[None] = 5.0
    W_curr[None][0] = -.5
    W_curr[None][1] = -0.1
    ymin[None] = -1.0
    ymax[None] = 1.0
    wscale[None] = 0.75
    L2w[None] = 200.0
    stepsize[None] = 0.0
    wvis_n[None] = 0
    
    # lmax[None] = 1.3

    for i, j in image:
        image[i,j] = ti.Vector([0.5, 0.5, 0.5])
    for i, j in ximage:
        ximage[i, j] = 0.0
    for i, j in ximage:
        wimage[i, j] = 0.0

# Create a GUI window
window = ti.ui.Window(
    name="Optimisation", res=(resolution*2, resolution), pos=(15, 50), fps_limit=100)

init_model(1)
init()
# X, Y = generate_training_samples(42, num_samples=X_train_num, r0=1.0, r1=2.0, r2=3.0)
X, Y = generate_training_samples2(
    seed=3, 
    num_samples=X_train_num, 
    num_clusters=4, xboundary=XBND[None]*0.8, std_dev=0.4)
X_all = X.copy()
Y_all = Y.copy()

compute_X_train_vis(X, Y)
L = compute_W_image(X, Y)
update_W_vis(True)
refresh_w_image = False
while window.running:
    # Compute the image using the kernel
    gui = window.get_gui()
    canvas = window.get_canvas()
    wscale[None] = gui.slider_float("wscale", wscale[None], minimum=0.01, maximum=2.0)
    L2w[None] = gui.slider_float("L2_W x e^-5", L2w[None], minimum=0.0, maximum=1000)
    W_curr[None][0] = gui.slider_float("W_curr 0", W_curr[None][0], minimum=-WBND[None], maximum=WBND[None])
    W_curr[None][1] = gui.slider_float("W_curr 1", W_curr[None][1], minimum=-WBND[None], maximum=WBND[None])
    stepsize[None] = gui.slider_float("LogStepsize", stepsize[None], minimum=-5, maximum=1.0)

    update_w_loss_pressed = gui.button("Update W-Loss Image")
    if update_w_loss_pressed:
        refresh_w_image = True

    if refresh_w_image:
        if batchmode:
            L = compute_W_image(X_batch, Y_batch)
        else:
            L = compute_W_image(X, Y)
        refresh_w_image = False

    if window.is_pressed('n'): # next step
        if batchmode:
            print('grad using batch')
            compute_W_grad(X_batch, Y_batch)
        else:
            compute_W_grad(X, Y)
        print(W_grad)
        update_W_curr()
        update_W_vis(True)

    if window.is_pressed('b'): # sample minibatch
        ii = np.random.choice(X_train_num, size=(batchsize,), replace=False)
        X_batch = X[ii]
        Y_batch = Y[ii]
        if not batchmode:
            batchmode = True

    if window.is_pressed('w'):
        refresh_w_image = True
    if window.is_pressed('r'):
        batchmode = False
        refresh_w_image = True


    set_out_w(W_curr[None]) 
    compute_X_image()
    #print(ximage.to_numpy()[:5, :5])
    fill_w_image()
    update_W_vis(False)
    
    img_np = image.to_numpy()
    canvas.set_image(img_np)
    canvas.circles(X_train_vis, radius=0.005, per_vertex_color=X_train_vis_clr)
    canvas.circles(W_curr_vis, radius=0.005, color=(0.0, 0.0, 0.5))
    if batchmode:
        compute_X_batch_vis(X_batch)
        canvas.circles(X_batch_vis, radius=0.003, color=(1., 1., 1.))
    
    window.show()
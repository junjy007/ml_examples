import taichi as ti
import numpy as np
import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam


DATADIR = os.path.expanduser("~/data/common")  # where mnist dataset is to be stored

class CustomTransform:
    def __call__(self, img):
        # Resize the image to 16x16 pixels
        img = F.resize(img, (16, 16))
        # Convert image to tensor
        img = F.to_tensor(img)
        # Binarize the image (threshold at 0.5)
        img = torch.round(img)
        # Flip the image upside down
        img = F.vflip(img)
        return img

def load_mnist_data():
    transform = CustomTransform()
    mnist_dataset = datasets.MNIST(root=DATADIR, train=True, download=True, transform=transform)

    # Filter out only 0 and 1 classes
    mnist_dataset.data = mnist_dataset.data[(mnist_dataset.targets == 0) | (mnist_dataset.targets == 1)]
    mnist_dataset.targets = mnist_dataset.targets[(mnist_dataset.targets == 0) | (mnist_dataset.targets == 1)]
    
    return mnist_dataset

mnist_dataset = load_mnist_data()
dl = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

current_image_index = 0

# Initialize Taichi with GUI
ti.init(arch=ti.cpu)

################################################################################
# Variables
################################################################################

# Network dimensions
n_input = 16
n_hidden = 4
n_output = 2
num_input_neurons = n_input * n_input  # 256

torch_model = nn.Sequential(
    nn.Linear(in_features=num_input_neurons, out_features=n_hidden, bias=False),
    nn.Tanh(),
    nn.Linear(n_hidden, n_output, bias=False)
)

# overall
mode_status = ti.field(ti.f32, shape=())

# Fields for input layer
V_input = ti.field(ti.f32, shape=(num_input_neurons,))
fixed_V_input = ti.field(ti.f32, shape=(num_input_neurons,))
positions_input = ti.Vector.field(2, dtype=ti.f32, shape=(num_input_neurons,))
colors_input = ti.Vector.field(3, dtype=ti.f32, shape=(num_input_neurons,))

# Fields for hidden layer
V_hidden = ti.field(ti.f32, shape=(n_hidden,))
fixed_V_hidden = ti.field(ti.f32, shape=(n_hidden,))
positions_hidden = ti.Vector.field(2, dtype=ti.f32, shape=(n_hidden,))
colors_hidden = ti.Vector.field(3, dtype=ti.f32, shape=(n_hidden,))

# Fields for output layer
z_output = ti.field(dtype=ti.f32, shape=(n_output,))
exp_z_output = ti.field(dtype=ti.f32, shape=(n_output,))
V_output = ti.field(ti.f32, shape=(n_output,))
fixed_V_output = ti.field(ti.f32, shape=(n_output,))
positions_output = ti.Vector.field(2, dtype=ti.f32, shape=(n_output,))
colors_output = ti.Vector.field(3, dtype=ti.f32, shape=(n_output,))

# Weight matrices
W_input_hidden = ti.field(dtype=ti.f32, shape=(n_hidden, num_input_neurons))
W_hidden_output = ti.field(dtype=ti.f32, shape=(n_output, n_hidden))

# Temperature and external field (if needed)
T = ti.field(ti.f32, shape=())
B = ti.field(ti.f32, shape=())

# Status of input data
display_X, display_y = None, None
positions_loss_bar = ti.Vector.field(2, dtype=ti.f32, shape=(2,))

# Optimisation
delta_output = ti.field(dtype=ti.f32, shape=(n_output,))
delta_hidden = ti.field(dtype=ti.f32, shape=(n_hidden,))
learning_rate = ti.field(dtype=ti.f32, shape=())
learning_rate[None] = 0.001  # You can adjust this value


# Initialize the network
@ti.kernel
def init():
    mode_status[None] = 0
    T[None] = 1
    B[None] = 0
    for i in range(num_input_neurons):
        fixed_V_input[i] = -100  # Indicates that the neuron is not fixed
    for i in range(n_hidden):
        fixed_V_hidden[i] = -100  # Indicates that the neuron is not fixed
    for i in range(n_output):
        fixed_V_output[i] = -100  # Indicates that the neuron is not fixed

    # Initialize weights with random values between -1 and 1
    for i in range(n_hidden):
        for j in range(num_input_neurons):
            W_input_hidden[i, j] = ti.random(ti.f32) * 2 - 1

    for i in range(n_output):
        for j in range(n_hidden):
            W_hidden_output[i, j] = ti.random(ti.f32) * 2 - 1

@ti.kernel
def initialize_neurons():
    # Input layer positions and activations
    for i in range(num_input_neurons):
        ix = i % n_input
        iy = i // n_input
        x = 0.05 + ix / (n_input - 1) * (0.5 - 0.05)
        y = 0.05 + iy / (n_input - 1) * (0.8 - 0.05)
        positions_input[i] = ti.Vector([x, y])
        V_input[i] = -1.0  # Initialize to -1.0
        fixed_V_input[i] = -999  # Not fixed

    # Hidden layer positions and activations
    for i in range(n_hidden):
        x = 0.6  # x position for hidden neurons
        y = 0.05 + i / (n_hidden - 1) * (0.8 - 0.05)
        positions_hidden[i] = ti.Vector([x, y])
        V_hidden[i] = 0.0  # Initialize activations

    # Output layer positions and activations
    for i in range(n_output):
        x = 0.8  # x position for output neurons
        y = 0.05 + i / (n_output - 1) * (0.8 - 0.05)
        positions_output[i] = ti.Vector([x, y])
        V_output[i] = 0.0  # Initialize activations


@ti.kernel
def compute_loss_vis(out_i:int):
    xmin = 0.81
    xmax = 0.98
    maxloss = -ti.log(1e-4)

    prob_pred = V_output[out_i]
    loss = -ti.log(max(prob_pred, 1e-4))
    barlen_ratio = loss / maxloss
    x1 = barlen_ratio * (xmax-xmin) + xmin

    y = 0.05 + out_i / (n_output - 1) * (0.8 - 0.05) # see `initialize_neurons`
    
    if mode_status[None] == 1:
        positions_loss_bar[0] = ti.Vector([xmin, y])
        positions_loss_bar[1] = ti.Vector([x1, y])
    
@ti.kernel
def optimise_step(label: int):
    # Step 1: Compute delta_output
    for i in range(n_output):
        y_i = 1.0 if i == label else 0.0
        delta_output[i] = V_output[i] - y_i  # Gradient of loss w.r.t z_i

    # Step 2: Compute delta_hidden
    for j in range(n_hidden):
        derivative_hidden = 1.0 - V_hidden[j] ** 2  # Derivative of tanh activation
        sum = 0.0
        for i in range(n_output):
            sum += delta_output[i] * W_hidden_output[i, j]
        delta_hidden[j] = sum * derivative_hidden

    # Step 3: Update weights between hidden and output layers
    for i in range(n_output):
        for j in range(n_hidden):
            grad = delta_output[i] * V_hidden[j]
            W_hidden_output[i, j] -= learning_rate[None] * grad

    # Step 4: Update weights between input and hidden layers
    for j in range(n_hidden):
        for k in range(num_input_neurons):
            grad = delta_hidden[j] * V_input[k]
            W_input_hidden[j, k] -= learning_rate[None] * grad

####
import pickle 
def save_weights():
    W1 = W_input_hidden.to_numpy()
    W2 = W_hidden_output.to_numpy()
    with open("nn3b.weights", "wb") as f:
        pickle.dump([W1, W2], f)
def load_weights():
    try:
        with open("nn3b.weights", "rb") as f:
            W1, W2 = pickle.load(f)
        W_input_hidden.from_numpy(W1)
        W_hidden_output.from_numpy(W2)
    except:
        print("load weights failed")

# Torch delegated training
def copy_weights_taichi_to_torch():
    # Copy W_input_hidden
    W1 = W_input_hidden.to_numpy()  # Shape: (n_hidden, num_input_neurons)
    W1_torch = torch.from_numpy(W1)
    torch_model[0].weight.data.copy_(W1_torch)
    
    # Copy W_hidden_output
    W2 = W_hidden_output.to_numpy()  # Shape: (n_output, n_hidden)
    W2_torch = torch.from_numpy(W2)
    torch_model[2].weight.data.copy_(W2_torch)

def copy_weights_torch_to_taichi():
    # Copy W_input_hidden
    W1_torch = torch_model[0].weight.data  # Shape: (n_hidden, num_input_neurons)
    W1 = W1_torch.numpy()
    W_input_hidden.from_numpy(W1)
    
    # Copy W_hidden_output
    W2_torch = torch_model[2].weight.data  # Shape: (n_output, n_hidden)
    W2 = W2_torch.numpy()
    W_hidden_output.from_numpy(W2)

def optimise_step_torch(steps:int=10):
    # Ensure the model is in training mode
    torch_model.train()
    
    # Copy weights from Taichi to PyTorch
    copy_weights_taichi_to_torch()
    optim = Adam(torch_model.parameters(), lr=1e-3)
    
    # Prepare the input data
    # input_data: torch tensor of shape (num_input_neurons,)
    # Reshape to (1, num_input_neurons) for batch size of 1
    step_num = 0
    for x, y in dl:
        input_tensor = x.view(y.shape[0], -1) * 2.0 - 1.0
    
        # Zero the gradients
        optim.zero_grad()
        
        # Forward pass
        output = torch_model(input_tensor)
        
        # Compute the loss
        # Using CrossEntropyLoss, which combines LogSoftmax and NLLLoss
        criterion = nn.CrossEntropyLoss()
        # target_label should be a tensor of shape (1,) with the class index
        # target_tensor = torch.tensor([target_label], dtype=torch.long)
        loss = criterion(output, y)
        
        # Backward pass and optimization
        loss.backward()
        print(loss.detach().item())
        optim.step()

        step_num += 1
        if step_num >= steps:
            break

    # Copy updated weights back to Taichi
    copy_weights_torch_to_taichi()

####

@ti.kernel
def update_input():
    for i in range(num_input_neurons):
        if fixed_V_input[i] > -10:
            V_input[i] = fixed_V_input[i]  # Use fixed value if set
        else:
            if mode_status[None] == 0:
                # Randomly choosen the input modes
                if ti.random(ti.f32) < 1/(1+ti.exp(-B[None])):
                    V_input[i] = 1.0
                else:
                    V_input[i] = -1.0

            elif mode_status[None] == 2: # mode - 1's inputs are fixed by images
                sum = 0.0
                for hidden_ind in range(n_hidden):
                    sum += W_input_hidden[hidden_ind, i] * V_hidden[hidden_ind]
                p = 1 / (1+ti.exp(-sum * 2.0 / T[None] - B[None]/T[None]))
                if ti.random(ti.f32) < p:
                    V_input[i] = 1.0
                else:
                    V_input[i] = -1.0




@ti.kernel
def update_hidden():
    if mode_status[None] == 0 or mode_status[None] == 1:
        for i in range(n_hidden):
            sum = 0.0
            for j in range(num_input_neurons):
                sum += W_input_hidden[i, j] * V_input[j]
            V_hidden[i] = ti.tanh(sum)  # Activation function
    elif mode_status[None] == 2:
        for i in range(n_hidden):
            if fixed_V_hidden[i] > -10:
                V_hidden[i] = fixed_V_hidden[i]
            else: # sample with Hopfield dynamics
                sum = 0.0
                # for j in range(num_input_neurons):
                    # sum += W_input_hidden[i, j] * V_input[j]
                for j in range(n_output):
                    sum += W_hidden_output[j, i] * (V_output[j] * 2.0 - 1.0)
                p = 1 / (1+ti.exp(-sum * 2.0 / T[None]))
                if ti.random(ti.f32) < p:
                    V_hidden[i] = 1.0
                else:
                    V_hidden[i] = -1.0
                

# @ti.kernel
# def update_output():
#     for i in range(n_output):
#         sum = 0.0
#         for j in range(n_hidden):
#             sum += W_hidden_output[i, j] * V_hidden[j]
#         V_output[i] = 1 / (1 + ti.exp(-sum))  # Activation function

@ti.kernel
def update_output():
    if mode_status[None] == 0 or mode_status[None] == 1:
        # Step 1: Compute weighted sums z_i
        for i in range(n_output):
            z_output[i] = 0.0
            for j in range(n_hidden):
                z_output[i] += W_hidden_output[i, j] * V_hidden[j]
        
        # Step 2: Compute exponentials and sum
        exp_sum = 0.0
        for i in range(n_output):
            exp_z_output[i] = ti.exp(z_output[i])
            exp_sum += exp_z_output[i]
        
        # Step 3: Compute softmax activations
        for i in range(n_output):
            V_output[i] = exp_z_output[i] / exp_sum
    elif mode_status[None] == 2:
        for i in range(n_output):
            if fixed_V_output[i] > -10:
                V_output[i] = fixed_V_output[i]
            else: # sample with Hopfield dynamics
                sum = 0.0
                for j in range(n_hidden):
                    sum += W_hidden_output[i, j] * V_hidden[j]
                p = 1 / (1+ti.exp(-sum * 2.0 / T[None]))
                if ti.random(ti.f32) < p:
                    V_output[i] = 1.0
                else:
                    V_output[i] = 0.0

@ti.kernel
def update_colors():
    # Update colors for input neurons
    for i in range(num_input_neurons):
        if V_input[i] > 0.0:
            colors_input[i] = ti.Vector([1.0, 1.0, 1.0])
        else:
            colors_input[i] = ti.Vector([0.0, 0.0, 0.0])

    # Update colors for hidden neurons
    for i in range(n_hidden):
        val = (V_hidden[i] + 1.0) / 2.0  # Map from [-1,1] to [0,1]
        colors_hidden[i] = ti.Vector([val, val, val])

    # Update colors for output neurons
    for i in range(n_output):
        val = V_output[i]
        colors_output[i] = ti.Vector([val, val, val])

################################################################################
# Helpers: manipulate states manually
################################################################################
@ti.kernel
def clear_fixed_states():
    for i in range(num_input_neurons):
        fixed_V_input[i] = -100  # Indicates that the neuron is not fixed
    for i in range(n_hidden):
        fixed_V_hidden[i] = -100  # Indicates that the neuron is not fixed
    for i in range(n_output):
        fixed_V_output[i] = -100  # Indicates that the neuron is not fixed


@ti.kernel
def set_pos(i: int):
    fixed_V_input[i] = 1.0
@ti.kernel
def set_neg(i: int):
    fixed_V_input[i] = -1.0
@ti.kernel
def set_free(i: int):
    fixed_V_input[i] = -100.0

@ti.kernel
def set_input(i: int):
    fixed_V_input.from_numpy(mnist_dataset[i])

@ti.kernel
def set_hid_pos(i: int):
    fixed_V_hidden[i] = 1.0

@ti.kernel
def set_hid_neg(i: int):
    fixed_V_hidden[i] = -1.0

@ti.kernel
def set_hid_free(i: int):
    fixed_V_hidden[i] = -100.0

@ti.kernel
def set_out_pos(i: int):
    fixed_V_output[i] = 1.0

@ti.kernel
def set_out_neg(i: int):
    fixed_V_output[i] = 0.0 # when using out, we take *2 -1, so its range is 0-1

@ti.kernel
def set_out_free(i: int):
    fixed_V_output[i] = -100.0


################################################################################
# Connect to image
################################################################################
def set_input_neurons_from_image(image_tensor):
    # image_tensor is a tensor of shape [1, 16, 16]
    image_array = image_tensor.squeeze().numpy().flatten()
    # Map pixel values from {0, 1} to {-1.0, 1.0}
    input_values = image_array.astype(np.float32) * 2.0 - 1.0
    fixed_V_input.from_numpy(input_values)

################################################################################
# Main Loop
################################################################################

window = ti.ui.Window(
    name="Feedforward Neural Network Visualization", res=(800, 512), pos=(15, 50), 
    fps_limit=30)

init()
initialize_neurons()
t = 0
train_counter = 0
while window.running:
    gui = window.get_gui()
    canvas = window.get_canvas()
    T[None] = ti.exp(gui.slider_float("Temperature", ti.log(T[None]), minimum=-10, maximum=5))
    B[None] = gui.slider_float("ExField", B[None], minimum=-10, maximum=10)
    current_image_index = gui.slider_int("Train Sample Id", current_image_index,
        minimum=0, maximum=1000)

    do_reset = gui.button("reset")
    if do_reset:
        init()
        initialize_neurons()


    if gui.button("Toggle input"):
        mode_status[None] = (mode_status[None] + 1) % 3
        clear_fixed_states()

    if mode_status[None] == 1:
        if gui.button("Save Weights"):
            save_weights()
        if gui.button("Load Weights"):
            load_weights()


        if window.is_pressed(ti.ui.LEFT):
            current_image_index -= 1
            if current_image_index<0:
                current_image_index=1000

            time.sleep(0.1)

        if window.is_pressed(ti.ui.RIGHT):
            current_image_index += 1
            if current_image_index>1000:
                current_image_index=0

            time.sleep(0.1)
        if train_counter == 0:
            display_X, display_y = mnist_dataset[current_image_index]
            fixed_V_input.from_torch((display_X.flatten()*2.0-1.0))
            if gui.button("optimise step"):
                optimise_step(display_y)

        elif train_counter > 0:
            train_counter -= 1
            
            # for X_batch, y_batch in dl:
                # mnist_dataset[np.random.randint(0, 500, size=(16,))]
            optimise_step_torch(100)
            
            # fixed_V_input.from_torch(X_batch[0].flatten())
        train_counter = gui.slider_int("train x steps", train_counter, 0, 50)


    if mode_status[None] == 0 or mode_status[None] == 2:
        if gui.button("Free all nodes"):
            clear_fixed_states()

        pos = window.get_cursor_pos()

        if pos[0] <= 0.55: # manage inputs
            xi = int((pos[0] - 0.05) / (0.5 - 0.05) * (n_input - 1) + 0.5)
            yi = int((pos[1] - 0.05) / (0.8 - 0.05) * (n_input - 1) + 0.5)
            if 0 <= xi < n_input and 0 <= yi < n_input:
                idx = yi * n_input + xi

                if window.get_event():
                    if window.event.key == "a":# ti.ui.LMB:
                        set_pos(idx)
                        print(yi, xi, idx, "+")
                        
                    elif window.event.key == "z":# ti.ui.RMB:
                        set_neg(idx)
                        print(yi, xi, idx, "-")

                    elif window.event.key == "q":# ti.ui.RMB:
                        set_free(idx)
                        print(yi, xi, idx, ".")

        elif pos[0] > 0.59 and pos[0] < 0.61:
            yi = int((pos[1] - 0.05) / (0.8 - 0.05) * (n_hidden - 1) + 0.5)
            if yi >=0 and yi < n_hidden:
                if window.get_event():
                    if window.event.key == "a":# ti.ui.LMB:
                        set_hid_pos(yi)
                        print(f"set hidden {yi} + ")
                    if window.event.key == "z":# ti.ui.LMB:
                        set_hid_neg(yi)
                        print(f"set hidden {yi} - ")
                    if window.event.key == "q":# ti.ui.LMB:
                        set_hid_free(yi)
                        print(f"set hidden {yi} free ")

        elif pos[0] > 0.78 and pos[0] < 0.82:
            yi = int((pos[1] - 0.05) / (0.8 - 0.05) * (n_output - 1) + 0.5)
            if yi >=0 and yi < n_output:
                if window.get_event():
                    if window.event.key == "a":# ti.ui.LMB:
                        set_out_pos(yi)
                        print(f"set output {yi} + ")
                    if window.event.key == "z":# ti.ui.LMB:
                        set_out_neg(yi)
                        print(f"set output {yi} - ")
                    if window.event.key == "q":# ti.ui.LMB:
                        set_out_free(yi)
                        print(f"set output {yi} free ")

                

    update_input()

    update_hidden()
    update_output()
    update_colors()

    # Draw neurons
    canvas.circles(positions_input, radius=0.01, per_vertex_color=colors_input)
    canvas.circles(positions_hidden, radius=0.01, color=(0.0, 0.3, 0.0))
    canvas.circles(positions_hidden, radius=0.008, per_vertex_color=colors_hidden)
    canvas.circles(positions_output, radius=0.015, color=(0.0, 0.3, 0.0))
    canvas.circles(positions_output, radius=0.012, per_vertex_color=colors_output)
    if mode_status[None] == 1:
        compute_loss_vis(display_y) # must be after update_hidden and output
        canvas.lines(positions_loss_bar, width=0.05, color=(1.0, 0.0, 0.0))

    window.show()
    t += 1
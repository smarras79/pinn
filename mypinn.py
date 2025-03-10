import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os


# Parameters
c     = 1.0 # Advection velocity
alpha = 0.0 # Diffusion coefficient (set to 0.0 for no diffusion)
x_min, x_max = 0.0, 2.0
t_min, t_max = 0.0, 2/math.pi #1.0
num_collocation_points = 1000
num_initial_points = 100
num_boundary_points = 100
epochs = 20000
learning_rate = 1e-3
num_time_steps = 10  # Number of time steps for output
#eqs = "advection"
eqs = "burgers"
lplot_exact = False

# Output directory
output_dir = "solution_images" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

# Neural Network

class PINN(nn.Module):
    """
    Let's break down that part of the code, which defines the neural network architecture within the PINN:
    
    1. class PINN(nn.Module):

    This line defines a Python class named PINN that inherits from nn.Module.
    In PyTorch, nn.Module is the base class for all neural network modules. By inheriting from it, PINN becomes a PyTorch neural network.

    2. def __init__(self):
    This is the constructor of the PINN class. It's called when you create an instance of the PINN network (e.g., model = PINN()).
    super(PINN, self).__init__() calls the constructor of the parent class nn.Module, which is essential for proper initialization.

    3. self.net = nn.Sequential(...):

    nn.Sequential is a PyTorch container that allows you to create a neural network by stacking layers in a sequential order.
    he layers defined inside nn.Sequential will be executed in the order they are added.

    4. nn.Linear(2, 20):

    This is a linear layer (also known as a fully connected layer).
    2 represents the number of input features. In our case, the inputs are x (spatial coordinate) and t (time), so there are two input features.
    20 represents the number of output features (neurons) in this layer.
    In essence, this layer performs a linear transformation: output = input * weight + bias, where weight is a 2x20 matrix and bias is a 20-element vector. This layer takes the 2 inputs and transforms them into 20 outputs.

    5. nn.Tanh():
    This is the hyperbolic tangent activation function.
    Activation functions introduce non-linearity into the network, allowing it to learn complex relationships. Without non-linear activation functions, the entire network would be equivalent to a single linear layer.
    Tanh squashes the output of the previous linear layer to the range [-1, 1].

    Same loging goes for the following linear and Tanh.
    
    Finally, 1 output feature. This is because we want the network to output a single value, u(x, t), which is the approximated solution to the advection equation.
"""
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Instantiate the network
model = PINN()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def myresidual(u, u_t, u_x, u_xx, eqs):
    if eqs == "advection":
        return u_t + c*u_x - alpha*u_xx
    elif eqs == "burgers":
        return u_t + u*u_x #- alpha*u_xx

# Loss function
def physics_informed_loss(u, x, t):
    u_x  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t  = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual = myresidual(u, u_t, u_x, u_xx, eqs) #u_t + c*u_x - alpha*u_xx
    return torch.mean(residual**2)

# Initial condition (e.g., u(x, 0) = sin(pi * x))
def initial_condition_loss(u_initial, x_initial, eqs):
    if eqs == "advection":
        u_true_initial = torch.sin(torch.pi * x_initial)
        return torch.mean((u_initial - u_true_initial)**2)
    elif eqs == "burgers":
        u_true_initial = torch.sin(torch.pi * x_initial) + 0.5
        return torch.mean((u_initial - u_true_initial)**2)

# Boundary condition (e.g., periodic boundaries)
def boundary_condition_loss(u_left, u_right):
    return torch.mean((u_left - u_right)**2)

# Exact solution (for comparison)
def exact_solution(x, t, c):
    return np.sin(np.pi * (x - c * t))

# Training data
x_collocation    = torch.rand(num_collocation_points, 1) * (x_max - x_min) + x_min
t_collocation    = torch.rand(num_collocation_points, 1) * (t_max - t_min) + t_min
x_initial        = torch.rand(num_initial_points, 1) * (x_max - x_min) + x_min
t_initial        = torch.zeros(num_initial_points, 1)
x_boundary_left  = torch.ones(num_boundary_points, 1) * x_min
x_boundary_right = torch.ones(num_boundary_points, 1) * x_max
t_boundary       = torch.rand(num_boundary_points, 1) * (t_max - t_min) + t_min

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Collocation loss
    x_collocation.requires_grad_(True)
    t_collocation.requires_grad_(True)
    u_collocation = model(x_collocation, t_collocation)
    loss_pde      = physics_informed_loss(u_collocation, x_collocation, t_collocation)
    
    # Initial condition loss
    u_initial_pred = model(x_initial, t_initial)
    loss_initial   = initial_condition_loss(u_initial_pred, x_initial, eqs)

    # Boundary condition loss
    u_boundary_left  = model(x_boundary_left, t_boundary)
    u_boundary_right = model(x_boundary_right, t_boundary)
    loss_boundary    = boundary_condition_loss(u_boundary_left, u_boundary_right)
    
    # Total loss
    loss = loss_pde + loss_initial + loss_boundary

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4e}")

# Outputting solution at time steps
x_plot = torch.linspace(x_min, x_max, 100).view(-1, 1)
time_steps = torch.linspace(t_min, t_max, num_time_steps)

for i, t_val in enumerate(time_steps):
    t_plot = torch.ones_like(x_plot) * t_val
    u_pred_plot = model(x_plot, t_plot).detach().numpy()
    x_np = x_plot.numpy().flatten()
    t_np = t_val.item()
    u_exact = exact_solution(x_np, t_np, c)

    plt.figure()
    plt.plot(x_plot.numpy(), u_pred_plot, label='PINN Solution')
    if lplot_exact == True:
        plt.plot(x_np, u_exact, '--', label='Exact Solution')
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title(f"Solution at t = {t_val.item():.2f}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"solution_t_{i:03d}epochs"+str(epochs)+".png"))
    plt.close()

print(f"Solution images saved in '{output_dir}'")

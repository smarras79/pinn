import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Viscosity constant
nu = 0.01 / np.pi

# Define the neural network
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.activation = nn.Tanh()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x, t):
        xt = torch.cat((x, t), dim=1)
        out = xt
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

# PDE residual
def pde_residual(model, x, t):
    x.requires_grad = True
    t.requires_grad = True
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx

# Gaussian initial condition
def gaussian_ic(x, A=1.0, x0=0.0, sigma=0.1):
    return A * torch.exp(-(x - x0)**2 / (2 * sigma**2))

# Define model
model = PINN([2, 40, 40, 40, 1]).to(device)

# Training data
N_f = 10000  # collocation points
N_i = 200    # initial condition points
N_b = 200    # boundary points

# Interior collocation points
x_f = torch.FloatTensor(N_f, 1).uniform_(-1, 1).to(device)
t_f = torch.FloatTensor(N_f, 1).uniform_(0, 1).to(device)

# Initial condition points
x_i = torch.linspace(-1, 1, N_i).view(-1, 1).to(device)
t_i = torch.zeros_like(x_i).to(device)
u_i = gaussian_ic(x_i).to(device)

# Boundary condition points
x_b = torch.cat([torch.ones(N_b//2, 1) * -1, torch.ones(N_b//2, 1)], dim=0).to(device)
t_b = torch.rand(N_b, 1).to(device)
u_b = torch.zeros_like(x_b).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
def train(epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()

        u_pred_i = model(x_i, t_i)
        u_pred_b = model(x_b, t_b)
        f_pred = pde_residual(model, x_f, t_f)

        loss_i = torch.mean((u_pred_i - u_i)**2)
        loss_b = torch.mean((u_pred_b - u_b)**2)
        loss_f = torch.mean(f_pred**2)

        loss = loss_i + loss_b + loss_f
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

train(3000)


import torch

# Parameters
nu = 0.01 / torch.pi  # viscosity
A = 1.0               # Gaussian amplitude
x0 = 0.0              # center of the Gaussian
sigma = 0.1           # standard deviation

# Integration settings
def exact_solution_torch(x, t, nx=200):
    """
    Computes the exact solution u(x, t) of the viscous Burgers equation
    with Gaussian initial condition using PyTorch.

    Arguments:
    x -- (N, 1) torch tensor, positions to evaluate
    t -- scalar time
    nx -- number of integration points

    Returns:
    u -- (N, 1) exact solution at each x
    """
    if t == 0:
        return A * torch.exp(-((x - x0)**2) / (2 * sigma**2))

    # Integration points
    y = torch.linspace(-10, 10, nx).view(1, -1).to(x.device)  # (1, nx)
    dy = y[0, 1] - y[0, 0]

    # Broadcasting to shape (N, nx)
    x = x.view(-1, 1)  # (N, 1)

    kernel = torch.exp(- (x - y)**2 / (4 * nu * t))           # heat kernel
    initial = torch.exp(- (y - x0)**2 / (2 * sigma**2))       # initial Gaussian

    exp_term = kernel * initial

    numerator = torch.sum((x - y) / (t + sigma**2 / (2 * nu)) * exp_term, dim=1, keepdim=True) * dy
    denominator = torch.sum(exp_term, dim=1, keepdim=True) * dy

    return numerator / denominator



# Sample x and time t
x_sample = torch.linspace(-1, 1, 100).view(-1, 1).to('cuda' if torch.cuda.is_available() else 'cpu')
t_sample = 0.5

# Compute exact solution
u_exact = exact_solution_torch(x_sample, t_sample)

# Optional: plot it
import matplotlib.pyplot as plt
plt.plot(x_sample.cpu().numpy(), u_exact.cpu().detach().numpy(), label="Exact")
plt.title(f"Exact solution at t = {t_sample}")
plt.xlabel("x"); plt.ylabel("u(x,t)")
plt.legend()
plt.grid()
plt.show()

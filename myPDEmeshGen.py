import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

class GridGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GridGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def laplace_loss(net, x_in, y_in):
    x_in.requires_grad_(True)
    y_in.requires_grad_(True)
    inputs = torch.stack([x_in, y_in], dim=1)
    outputs = net(inputs)
    x = outputs[:, 0].unsqueeze(1)
    y = outputs[:, 1].unsqueeze(1)

    x_xx = torch.autograd.grad(torch.autograd.grad(x, x_in, grad_outputs=torch.ones_like(x), create_graph=True)[0], x_in, grad_outputs=torch.ones_like(x_in), create_graph=True)[0]
    x_yy = torch.autograd.grad(torch.autograd.grad(x, y_in, grad_outputs=torch.ones_like(x), create_graph=True)[0], y_in, grad_outputs=torch.ones_like(y_in), create_graph=True)[0]

    y_xx = torch.autograd.grad(torch.autograd.grad(y, x_in, grad_outputs=torch.ones_like(y), create_graph=True)[0], x_in, grad_outputs=torch.ones_like(x_in), create_graph=True)[0]
    y_yy = torch.autograd.grad(torch.autograd.grad(y, y_in, grad_outputs=torch.ones_like(y), create_graph=True)[0], y_in, grad_outputs=torch.ones_like(y_in), create_graph=True)[0]

    laplace_x = x_xx + x_yy
    laplace_y = y_xx + y_yy

    return torch.mean(laplace_x**2 + laplace_y**2)

def boundary_loss(net, x_in, y_in):
    inputs = torch.stack([x_in, y_in], dim=1)
    outputs = net(inputs)
    x = outputs[:, 0]
    y = outputs[:, 1]

    left_mask = torch.isclose(x_in, torch.tensor(0.0, dtype=torch.float32))
    right_mask = torch.isclose(x_in, torch.tensor(1.0, dtype=torch.float32))
    bottom_mask = torch.isclose(y_in, torch.tensor(0.0, dtype=torch.float32))
    top_mask = torch.isclose(y_in, torch.tensor(1.0, dtype=torch.float32))

    left_loss = torch.mean((x[left_mask] - 0.0)**2 + (y[left_mask] - y_in[left_mask])**2)
    right_loss = torch.mean((x[right_mask] - 1.0)**2 + (y[right_mask] - y_in[right_mask])**2)
    bottom_loss = torch.mean((x[bottom_mask] - x_in[bottom_mask])**2 + (y[bottom_mask] - 0.0)**2)
    top_loss = torch.mean((x[top_mask] - x_in[top_mask])**2 + (y[top_mask] - 1.0)**2)

    return left_loss + right_loss + bottom_loss + top_loss

# Parameters
input_dim = 2
hidden_dim = 20
output_dim = 2
num_points = 1000
learning_rate = 0.001
epochs = 5000
plot_interval = 200

# Network and optimizer
net = GridGenerator(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Training data
x_in = torch.rand(num_points)
y_in = torch.rand(num_points)

# Boundary points
boundary_points = 200
x_left = torch.zeros(boundary_points)
y_left = torch.rand(boundary_points)
x_right = torch.ones(boundary_points)
y_right = torch.rand(boundary_points)
x_bottom = torch.rand(boundary_points)
y_bottom = torch.zeros(boundary_points)
x_top = torch.rand(boundary_points)
y_top = torch.ones(boundary_points)

x_in = torch.cat([x_in, x_left, x_right, x_bottom, x_top])
y_in = torch.cat([y_in, y_left, y_right, y_bottom, y_top])

# Generate structured grid for plotting
x_grid = torch.linspace(0, 1, 10)
y_grid = torch.linspace(0, 1, 10)
x_mesh, y_mesh = torch.meshgrid(x_grid, y_grid, indexing='xy')
x_flat = x_mesh.flatten()
y_flat = y_mesh.flatten()
inputs_grid = torch.stack([x_flat, y_flat], dim=1)

# Create directory for saving plots
if not os.path.exists("square_grid_plots"):
    os.makedirs("square_grid_plots")

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    physics_loss_val = laplace_loss(net, x_in, y_in)
    boundary_loss_val = boundary_loss(net, x_in, y_in)
    loss = physics_loss_val + boundary_loss_val
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    if epoch % plot_interval == 0:
        grid_output = net(inputs_grid)
        x_out_grid = grid_output[:, 0].detach().numpy().reshape(x_mesh.shape)
        y_out_grid = grid_output[:, 1].detach().numpy().reshape(y_mesh.shape)

        plt.figure()
        plt.plot(x_out_grid, y_out_grid, 'b-')
        plt.plot(x_out_grid.T, y_out_grid.T, 'b-')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Epoch {epoch}")
        plt.savefig(f"square_grid_plots/epoch_{epoch}.png")
        plt.close()

# Generate and show final grid
grid_output = net(inputs_grid)
x_out_grid = grid_output[:, 0].detach().numpy().reshape(x_mesh.shape)
y_out_grid = grid_output[:, 1].detach().numpy().reshape(y_mesh.shape)

plt.figure()
plt.plot(x_out_grid, y_out_grid, 'b-')
plt.plot(x_out_grid.T, y_out_grid.T, 'b-')
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Final Grid (Square)")
plt.show()

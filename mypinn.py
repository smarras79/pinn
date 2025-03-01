import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Parameters
c = 1.0  # Advection velocity
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0
num_collocation_points = 1000
num_initial_points = 100
num_boundary_points = 100
epochs = 1000
learning_rate = 1e-3

# Neural Network
class PINN(nn.Module):
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

# Loss function
def physics_informed_loss(u, x, t):
    u_xt = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u_xt, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u_xt, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual = u_t + c * u_x
    return torch.mean(residual**2)

# Initial condition (e.g., u(x, 0) = sin(pi * x))
def initial_condition_loss(u_initial, x_initial):
    u_true_initial = torch.sin(torch.pi * x_initial)
    return torch.mean((u_initial - u_true_initial)**2)

# Boundary condition (e.g., periodic boundaries)
def boundary_condition_loss(u_left, u_right):
    return torch.mean((u_left - u_right)**2)

# Training data
x_collocation = torch.rand(num_collocation_points, 1) * (x_max - x_min) + x_min
t_collocation = torch.rand(num_collocation_points, 1) * (t_max - t_min) + t_min
x_initial = torch.rand(num_initial_points, 1) * (x_max - x_min) + x_min
t_initial = torch.zeros(num_initial_points, 1)
x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
x_boundary_right = torch.ones(num_boundary_points, 1) * x_max
t_boundary = torch.rand(num_boundary_points, 1) * (t_max - t_min) + t_min

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()

    # Collocation loss
    x_collocation.requires_grad_(True)
    t_collocation.requires_grad_(True)
    u_collocation = model(x_collocation, t_collocation)
    loss_pde = physics_informed_loss(u_collocation, x_collocation, t_collocation)

    # Initial condition loss
    u_initial_pred = model(x_initial, t_initial)
    loss_initial = initial_condition_loss(u_initial_pred, x_initial)

    # Boundary condition loss
    u_boundary_left = model(x_boundary_left, t_boundary)
    u_boundary_right = model(x_boundary_right, t_boundary)
    loss_boundary = boundary_condition_loss(u_boundary_left, u_boundary_right)

    # Total loss
    loss = loss_pde + loss_initial + loss_boundary

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4e}")

# Inference
x_test = torch.linspace(x_min, x_max, 100).view(-1, 1)
t_test = torch.ones_like(x_test) * 0.5 #Example time.
u_pred = model(x_test, t_test)

print("Predicted u values:", u_pred.detach().numpy())

# Inference for initial condition plot
x_initial_plot = torch.linspace(x_min, x_max, 100).view(-1, 1)
t_initial_plot = torch.zeros_like(x_initial_plot)
u_initial_pred_plot = model(x_initial_plot, t_initial_plot).detach().numpy()
u_true_initial_plot = np.sin(np.pi * x_initial_plot.numpy())

# Inference for final solution plot
x_final_plot = torch.linspace(x_min, x_max, 100).view(-1, 1)
t_final_plot = torch.ones_like(x_final_plot) * t_max
u_final_pred_plot = model(x_final_plot, t_final_plot).detach().numpy()

# Plotting
plt.figure(figsize=(12, 5))

# Initial condition plot
plt.subplot(1, 2, 1)
plt.plot(x_initial_plot.numpy(), u_initial_pred_plot, label="PINN Initial")
plt.plot(x_initial_plot.numpy(), u_true_initial_plot, '--', label="True Initial")
plt.xlabel("x")
plt.ylabel("u(x, 0)")
plt.title("Initial Condition")
plt.legend()

# Final solution plot
plt.subplot(1, 2, 2)
plt.plot(x_final_plot.numpy(), u_final_pred_plot, label="PINN Final")
plt.xlabel("x")
plt.ylabel(f"u(x, {t_max})")
plt.title("Final Solution")
plt.legend()

plt.tight_layout()
plt.show()

import torch 
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Parameters for Shallow Water Equations
g = 9.81        # Gravitational acceleration (m/s^2)
# Domain parameters
x_min, x_max = 0.0, 20.0    # Spatial domain [m]
L = x_max - x_min
# Training parameters
num_initial_points = 1500 
num_boundary_points = 200
epochs = 10000
num_collocation_points = 6000
learning_rate = 1e-3
num_time_steps = 20
#Scheduler tuning parameters
scheduler_step_size_frequency = 2 #Number of times we want scheduler to reduce LR during full training with epochs
scheduler_step_size = epochs // scheduler_step_size_frequency # Epoch intervals at which scheduler will reduce LR 
scheduler_gamma=0.5 #Factor by which scheduler will reduce LR at each epoch interval
# Initial condition parameters
eta_val = 0.33
q_val = 0.18
#Weights
lambda_c = 1.0
lambda_m = 10.0  # Increase if momentum is underfitting

# Output directory
output_dir = "swe/temp/swe_solution_oneInput_case6_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

def get_weights(epoch, total_epochs):
    pde_weight = 20.0
    bc_weight = 10.0
    return pde_weight, bc_weight

# ------------------ Input Normalization ------------------
def normalize(x, xmin=0.0, xmax=20.0):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

class ImprovedPINN_SWE(nn.Module):
    """
    Improved Physics-Informed Neural Network for 1D Shallow Water Equations
    """
    def __init__(self):
        super(ImprovedPINN_SWE, self).__init__()
        # Start changing the structure of the PINN to simplify it, clean IC and everything related to time
        # Remove any wet/dry --> this is only for dam break
        # Fewer layers, remove heads
        
        # same head for h and u
        self.hu_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Normalize inputs: Neural networks train better with inputs in the range [-1, 1]
        x_norm = normalize(x)
        inputs = torch.cat([x_norm], dim=1)
        h_raw = self.hu_head(inputs)
        u_raw = self.hu_head(inputs)
        return h_raw, u_raw


# Instantiate the network
model = ImprovedPINN_SWE()

# Optimizer with scheduled learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)

def improved_physics_loss(h, u, x):
    """
    Physics-informed loss for 1D Shallow Water Equations with gradient-based weighting.
    Penalizes discontinuity regions less and focuses on smooth wave dynamics.
    """
    q = h * u
    hu2 = h * u**2
    pressure = 0.5 * g * h**2

    # Compute derivatives
    hu_x = torch.autograd.grad(q, x, grad_outputs=torch.ones_like(q), create_graph=True)[0]
    flux_x = torch.autograd.grad(hu2 + pressure, x, grad_outputs=torch.ones_like(hu2), create_graph=True)[0]
    zb = bed_elevation(x)

    # Steady continuity residual: ∂(hu)/∂x = 0
    continuity_residual = hu_x

    # Momentum residual: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0 (only in wet regions)
    # Change the momentum residual to include the bed slope ∂zb/∂x
    #epsilon = 1e-6 #Avoids large or exploding gradients when h → 0 (common near wet-dry interfaces or sharp dam fronts)
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb), create_graph=True)[0]
    momentum_residual = flux_x + g * h * dzb_dx

    # Weighted PDE residuals
    continuity_loss = torch.mean(continuity_residual**2)
    momentum_loss = torch.mean(momentum_residual**2)

    # Combine total PDE loss
    total_pde_loss = continuity_loss + momentum_loss  

    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item()
    }

# ------------------ Bed Elevation Function ------------------

def bed_elevation(x: torch.Tensor) -> torch.Tensor:
    zb = torch.zeros_like(x)
    zb_h = 0.2 - 0.05 * (x - 10.0) **2
    return torch.where((x > 8.0) & (x < 12.0), zb_h, zb)

def set_eta_q(case=6):
    if case == 6:
        eta_val, q_val = 0.33, 0.18
    elif case == 7:
        eta_val, q_val = 2.0, 4.42
    else:
        raise ValueError("Invalid case")

# ------------------ Boundary Conditions ------------------
def eta_left():
    return torch.tensor([[eta_val]], dtype=torch.float32)

def eta_right():
    return torch.tensor([[eta_val]], dtype=torch.float32)

def h_bc_left():
    return eta_val - bed_elevation(torch.tensor([[x_min]]))

def h_bc_right():
    return eta_val - bed_elevation(torch.tensor([[x_max]]))

def u_bc_left():
    h = eta_left()
    q = torch.tensor([[q_val]])
    u = q / h
    return u

def u_bc_right():
    h = eta_right()
    q = torch.tensor([[q_val]])
    u = q / h
    return u

def boundary_condition_loss(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    """
    Simple outflow boundary conditions
    """

    loss_bc_left = torch.mean((h_left_pred - h_bc_left())**2) + torch.mean((u_left_pred - u_bc_left())**2)
    loss_bc_right = torch.mean((h_right_pred - h_bc_right())**2) + torch.mean((u_right_pred - u_bc_right())**2)
    return loss_bc_left + loss_bc_right

c0 = np.sqrt(g * eta_val)
# Generate training data
# Collocation points
x_collocation = torch.rand(int(num_collocation_points), 1) * (x_max - x_min) + x_min
# Boundary points
x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
x_boundary_right = torch.ones(num_boundary_points, 1) * x_max

# Set requires_grad
x_collocation.requires_grad_(True)
x_boundary_left.requires_grad_(True)
x_boundary_right.requires_grad_(True)

print(f"Domain: x ∈ [{x_min}, {x_max}]")

start_time = time.time()
loss_history = []

model.train()

# Loss weights
lambda_c = 1.0
lambda_m = 10.0

#Setting test case here. 
set_eta_q(6)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Physics loss
    h_collocation, u_collocation = model(x_collocation)
    loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                    x_collocation)
  
    # Boundary loss
    h_boundary_left, u_boundary_left = model(x_boundary_left)
    h_boundary_right, u_boundary_right = model(x_boundary_right)
    loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                          h_boundary_right, u_boundary_right)

    lambda_pde_curr, lambda_bc_curr = get_weights(epoch, epochs)

    # Total loss
    loss = (
        lambda_pde_curr * loss_pde
        + lambda_bc_curr * loss_boundary
    )

    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {loss.item():.4e}")
        print(f"  PDE Loss: {loss_pde.item():.4e}")
        print(f"  Boundary Loss: {loss_boundary.item():.4e}")
        print(f"  Continuity: {pde_components['continuity']:.4e}")
        print(f"  Momentum: {pde_components['momentum']:.4e}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Generate solution plots
x_plot = torch.linspace(x_min, x_max, 500).view(-1, 1)

print("\nDiagnostic check: did the model learn anything...")
with torch.no_grad():
    h_pred, u_pred = model(x_plot)
    print("Mean h:", h_pred.mean().item(), "Std h:", h_pred.std().item())
    print("Mean u:", u_pred.mean().item(), "Std u:", u_pred.std().item())

print("\nChecking model for time steps...")
with torch.no_grad():
    h_pred, u_pred = model(x_plot)
    h_pred_plot = h_pred.numpy().flatten()
    u_pred_plot = u_pred.numpy().flatten()
    # Compute bed elevation and free surface
    zb_plot = bed_elevation(x_plot)
    eta_plot = h_pred_plot + zb_plot.numpy().flatten() # Free surface
    x_np = x_plot.detach().numpy().flatten()
    
    # Get analytical solution
    #h_exact, u_exact = exact_dam_break_solution(x_np, t_np)
    
    # Calculate errors
    # h_error = np.abs(h_pred_plot - h_exact)
    # u_error = np.abs(u_pred_plot - u_exact)
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Water height
    ax1.plot(x_np, h_pred_plot, 'b-', label='PINN h(x,t)', linewidth=2)
    ax1.plot(x_np, eta_plot, 'm--', label='Free surface η = h + zb', linewidth=2)
    ax1.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
    #ax1.plot(x_np, h_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    #ax1.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax1.set_ylabel('Water Height h(x,t) [m]')
    ax1.set_title(f'Water Height')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, eta_val * 2.0)
    
    # Velocity
    ax2.plot(x_np, u_pred_plot, 'g-', label='PINN u(x,t)', linewidth=2)
    #ax2.plot(x_np, u_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    #ax2.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax2.set_ylabel('Velocity u(x,t) [m/s]')
    ax2.set_title(f'Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Height error
    #ax3.semilogy(x_np, np.maximum(h_error, 1e-10), 'r-', linewidth=2)
    #ax3.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax3.set_ylabel('|h_pred - h_exact|')
    #ax3.set_title(f'Height Error - Max: {np.max(h_error):.4f}')
    ax3.grid(True, alpha=0.3)
    
    # Velocity error
    #ax4.semilogy(x_np, np.maximum(u_error, 1e-10), 'g-', linewidth=2)
    #ax4.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Position x [m]')
    ax4.set_ylabel('|u_pred - u_exact|')
    #ax4.set_title(f'Velocity Error - Max: {np.max(u_error):.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add information text
    info_text = f"Epochs: {epochs} | Points: {num_collocation_points}\n"
    info_text += f"Training time: {elapsed_time:.1f}s | Final loss: {loss.item():.4e}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9)
    
    plt.savefig(os.path.join(output_dir, f"swe_solution.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Loss history plot
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss History')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'loss_history.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nSolution images saved in '{output_dir}'")
torch.save(model.state_dict(), os.path.join(output_dir, 'curriculum_swe_pinn_model.pth'))
print("Model saved successfully!")
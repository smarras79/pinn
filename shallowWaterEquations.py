import torch 
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Parameters for Shallow Water Equations
g = 1.0        # Gravitational acceleration (m/s^2)
H0 = 0.25        # Mean water depth (m)

# Domain parameters
x_min, x_max = -np.pi, np.pi    # Spatial domain [m]
t_min, t_max = 0.0, 1.0     # Time domain [s]
L = x_max - x_min

# Loss function weights - Dramatically increased IC weight
lambda_ic = 10000.0  # Much higher weight for initial conditions
lambda_pde = 1.0
lambda_bc = 100.0   # Increased weight for boundary conditions

# Training parameters
num_initial_points = 2000  # More initial condition points
num_boundary_points = 400
epochs = 2000  # More epochs for better convergence
num_collocation_points = 15000  # More collocation points
learning_rate = 1e-3
num_time_steps = 20

# Initial condition parameters
dam_position = -0.25          # Position of dam break
h_left = 0.25               # Water height on left side of dam
h_right = 0.0              # True dry bed condition
u_left = 0.0               # Initial velocity on left
u_right = 0.0              # Initial velocity on right

# Output directory
output_dir = "swe_solution_images_improved_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

# Improved Neural Network for Shallow Water Equations
class PINN_SWE(nn.Module):
    """
    Physics-Informed Neural Network for 1D Shallow Water Equations
    Enhanced architecture with better initialization and conditioning
    """
    def __init__(self):
        super(PINN_SWE, self).__init__()
        
        # Deeper network with residual-like connections
        self.net_h = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.net_u = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights with smaller values for better training
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        h_raw = self.net_h(inputs)
        u_raw = self.net_u(inputs)
        
        # Ensure h is non-negative and apply initial condition constraint
        h = torch.relu(h_raw)
        
        # Apply soft constraint for initial conditions when t is small
        t_threshold = 0.01
        ic_weight = torch.exp(-t / t_threshold)  # Strong constraint at t=0, weaker as t increases
        
        # Initial condition enforcement
        h_ic = torch.where(x < dam_position, 
                          torch.tensor(h_left, dtype=x.dtype, device=x.device), 
                          torch.tensor(0.0, dtype=x.dtype, device=x.device))
        u_ic = torch.zeros_like(u_raw)
        
        # Blend initial conditions with network output
        h = ic_weight * h_ic + (1 - ic_weight) * h
        u = ic_weight * u_ic + (1 - ic_weight) * u_raw
        
        return h, u

# Instantiate the network
model = PINN_SWE()

# Improved optimizer setup
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.7, min_lr=1e-6)

def physics_informed_loss(h, u, x, t):
    """
    Compute the physics-informed loss based on SWE residuals
    """
    # Compute gradients
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Compute (hu)_x for continuity equation
    hu = h * u
    hu_x = torch.autograd.grad(hu, x, grad_outputs=torch.ones_like(hu), create_graph=True)[0]
    
    # Continuity equation: ∂h/∂t + ∂(hu)/∂x = 0
    continuity_residual = h_t + hu_x
    
    # Momentum equation with wet/dry treatment
    wet_threshold = 1e-4
    wet_mask = (h > wet_threshold).float()
    
    # For wet regions: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0
    momentum_residual = u_t + u * u_x + g * h_x
    
    # For dry regions: u should be zero
    dry_velocity_residual = u * (1 - wet_mask)
    
    # Weighted losses
    loss_continuity = torch.mean(continuity_residual**2)
    loss_momentum = torch.mean((wet_mask * momentum_residual)**2)
    loss_dry = torch.mean(dry_velocity_residual**2)
    
    return loss_continuity + loss_momentum + 10.0 * loss_dry

def initial_condition_loss(h_pred, u_pred, x_initial):
    """
    Strict initial condition enforcement for dam break
    """
    # Exact initial conditions
    h_true = torch.where(x_initial < dam_position, 
                        torch.tensor(h_left, dtype=h_pred.dtype, device=h_pred.device), 
                        torch.tensor(0.0, dtype=h_pred.dtype, device=h_pred.device))
    u_true = torch.zeros_like(u_pred)
    
    # Higher weights near dam position for sharp transition
    dam_distance = torch.abs(x_initial - dam_position)
    weight_h = torch.where(dam_distance < 0.1, 
                          torch.tensor(100.0, dtype=x_initial.dtype, device=x_initial.device),
                          torch.tensor(1.0, dtype=x_initial.dtype, device=x_initial.device))
    
    # Separate losses for left and right regions
    left_mask = (x_initial < dam_position).float()
    right_mask = (x_initial >= dam_position).float()
    
    loss_h_left = torch.mean(left_mask * weight_h * (h_pred - h_true)**2)
    loss_h_right = torch.mean(right_mask * 1000.0 * h_pred**2)  # Force h=0 on right
    loss_u = torch.mean((u_pred - u_true)**2)
    
    return loss_h_left + loss_h_right + loss_u

def boundary_condition_loss(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    """
    Improved boundary conditions for dam break
    """
    # Outflow boundary conditions - minimal reflection
    loss_bc = 0.01 * (torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2))
    return loss_bc

def exact_dam_break_solution(x, t):
    """
    Analytical solution for dam break (Ritter's solution)
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(t, torch.Tensor):
        t = t.item()
    
    # At t=0, return exact initial conditions
    if t <= 1e-8:
        h_exact = np.where(x < dam_position, h_left, 0.0)
        u_exact = np.zeros_like(x)
        return h_exact, u_exact
    
    # Ritter's solution parameters
    c0 = np.sqrt(g * h_left)
    
    # Characteristic positions
    x_left = dam_position - c0 * t
    x_right = dam_position + 2 * c0 * t
    
    h_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    
    # Left region (undisturbed)
    mask_left = x <= x_left
    h_exact[mask_left] = h_left
    u_exact[mask_left] = 0.0
    
    # Rarefaction fan region
    mask_fan = (x > x_left) & (x < x_right)
    #if np.any(mask_fan):
    xi = (x[mask_fan] - dam_position) / t
    h_exact[mask_fan] = (1.0 / (9.0 * g)) * (2 * c0 - xi)**2
    u_exact[mask_fan] = (2.0 / 3.0) * (xi + c0)
    
    # Right region (dry bed)
    mask_right = x >= x_right
    h_exact[mask_right] = 0.0
    u_exact[mask_right] = 0.0
    
    return h_exact, u_exact

# Enhanced training data sampling
# Much denser sampling near dam break
x_dam_region = torch.linspace(dam_position - 0.3, dam_position + 0.3, num_collocation_points // 2)
x_outer_left = torch.linspace(x_min, dam_position - 0.3, num_collocation_points // 4)
x_outer_right = torch.linspace(dam_position + 0.3, x_max, num_collocation_points // 4)
x_collocation = torch.cat([x_outer_left, x_dam_region, x_outer_right]).reshape(-1, 1)

# Time sampling with more points near t=0
t_early = torch.rand(num_collocation_points // 2, 1) * 0.1  # More points near t=0
t_later = torch.rand(num_collocation_points // 2, 1) * 0.9 + 0.1
t_collocation = torch.cat([t_early, t_later])

# Much denser initial condition sampling near dam
x_ic_dense = torch.linspace(dam_position - 0.2, dam_position + 0.2, num_initial_points // 2)
x_ic_sparse = torch.cat([
    torch.linspace(x_min, dam_position - 0.2, num_initial_points // 4),
    torch.linspace(dam_position + 0.2, x_max, num_initial_points // 4)
])
x_initial = torch.cat([x_ic_dense, x_ic_sparse]).reshape(-1, 1)
t_initial = torch.zeros(num_initial_points, 1)

# Boundary points
x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
x_boundary_right = torch.ones(num_boundary_points, 1) * x_max
t_boundary = torch.rand(num_boundary_points, 1) * (t_max - t_min) + t_min

# Set requires_grad for automatic differentiation
x_collocation.requires_grad_(True)
t_collocation.requires_grad_(True)
x_boundary_left.requires_grad_(True)
x_boundary_right.requires_grad_(True)

start_time = time.time()

print("Starting improved training for 1D Shallow Water Equations...")
print(f"Domain: x ∈ [{x_min}, {x_max}], t ∈ [{t_min}, {t_max}]")
print(f"Dam break at x = {dam_position}, h_left = {h_left}, h_right = {h_right}")
print(f"Initial condition weight: {lambda_ic}")

# Training loop with additional monitoring
loss_history = []
ic_loss_history = []
pde_loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Initial condition loss - very strict enforcement
    h_initial_pred, u_initial_pred = model(x_initial, t_initial)
    loss_initial = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial)

    # Collocation (PDE) loss
    h_collocation, u_collocation = model(x_collocation, t_collocation)
    loss_pde = physics_informed_loss(h_collocation, u_collocation, x_collocation, t_collocation)
    
    # Boundary condition loss
    h_boundary_left, u_boundary_left = model(x_boundary_left, t_boundary)
    h_boundary_right, u_boundary_right = model(x_boundary_right, t_boundary)
    loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                          h_boundary_right, u_boundary_right)

    # Total loss with adaptive weighting
    adaptive_ic_weight = lambda_ic if epoch < epochs // 2 else lambda_ic * 0.5
    loss = loss_initial * adaptive_ic_weight + loss_pde * lambda_pde + loss_boundary * lambda_bc
    
    # Backpropagation
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step(loss)
    
    # Record losses
    loss_history.append(loss.item())
    ic_loss_history.append(loss_initial.item())
    pde_loss_history.append(loss_pde.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss.item():.4e}")
        print(f"  Initial condition loss: {loss_initial.item():.5e}")
        print(f"  PDE loss: {loss_pde.item():.5e}")
        print(f"  Boundary condition loss: {loss_boundary.item():.5e}")
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")

elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Generate solution plots with error analysis
x_plot = torch.linspace(x_min, x_max, 800).view(-1, 1)  # Higher resolution
time_steps = torch.linspace(t_min, t_max, num_time_steps)

for i, t_val in enumerate(time_steps):
    t_plot = torch.ones_like(x_plot) * t_val
    
    with torch.no_grad():
        h_pred, u_pred = model(x_plot, t_plot)
        h_pred_plot = h_pred.numpy().flatten()
        u_pred_plot = u_pred.numpy().flatten()
    
    x_np = x_plot.numpy().flatten()
    t_np = t_val.item()
    
    # Get analytical solution
    h_exact, u_exact = exact_dam_break_solution(x_np, t_np)
    
    # Calculate errors
    h_error = np.abs(h_pred_plot - h_exact)
    u_error = np.abs(u_pred_plot - u_exact)
    
    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot water height
    ax1.plot(x_np, h_pred_plot, 'b-', label='PINN h(x,t)', linewidth=2)
    ax1.plot(x_np, h_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    ax1.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax1.set_ylabel('Water Height h(x,t) [m]')
    ax1.set_title(f'Water Height - t = {t_np:.3f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, h_left * 1.1)
    
    # Plot velocity
    ax2.plot(x_np, u_pred_plot, 'g-', label='PINN u(x,t)', linewidth=2)
    ax2.plot(x_np, u_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    ax2.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax2.set_ylabel('Velocity u(x,t) [m/s]')
    ax2.set_title(f'Velocity - t = {t_np:.3f}s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot height error
    ax3.plot(x_np, h_error, 'r-', linewidth=2)
    ax3.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax3.set_ylabel('|h_pred - h_exact|')
    ax3.set_title(f'Height Error - Max: {np.max(h_error):.4f}')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot velocity error
    ax4.plot(x_np, u_error, 'g-', linewidth=2)
    ax4.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Position x [m]')
    ax4.set_ylabel('|u_pred - u_exact|')
    ax4.set_title(f'Velocity Error - Max: {np.max(u_error):.4f}')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    
    # Add information text
    info_text = f"Epochs: {epochs} | IC Weight: {lambda_ic} | Points: {num_collocation_points}\n"
    info_text += f"Training time: {elapsed_time:.1f}s | Final loss: {loss.item():.4e}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9)
    
    plt.savefig(os.path.join(output_dir, f"swe_solution_t_{i:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Enhanced loss history plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

ax1.semilogy(loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss')
ax1.set_title('Total Training Loss History')
ax1.grid(True, alpha=0.3)

ax2.semilogy(ic_loss_history, 'r-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Initial Condition Loss')
ax2.set_title('Initial Condition Loss History')
ax2.grid(True, alpha=0.3)

ax3.semilogy(pde_loss_history, 'g-', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('PDE Loss')
ax3.set_title('PDE Loss History')
ax3.grid(True, alpha=0.3)

# Learning rate history
lr_history = [group['lr'] for group in optimizer.param_groups for _ in range(len(loss_history))]
ax4.semilogy(lr_history[:len(loss_history)], 'm-', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_title('Learning Rate Schedule')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comprehensive_loss_history.png'), dpi=150, bbox_inches='tight')
plt.close()

# Test initial condition match
print("\n" + "="*60)
print("INITIAL CONDITION VALIDATION")
print("="*60)

x_test = torch.linspace(x_min, x_max, 100).view(-1, 1)
t_test = torch.zeros_like(x_test)

with torch.no_grad():
    h_test, u_test = model(x_test, t_test)
    h_test_np = h_test.numpy().flatten()
    u_test_np = u_test.numpy().flatten()

x_test_np = x_test.numpy().flatten()
h_exact_test, u_exact_test = exact_dam_break_solution(x_test_np, 0.0)

h_ic_error = np.mean(np.abs(h_test_np - h_exact_test))
u_ic_error = np.mean(np.abs(u_test_np - u_exact_test))

print(f"Initial condition errors:")
print(f"  Water height MAE: {h_ic_error:.6f}")
print(f"  Velocity MAE: {u_ic_error:.6f}")
print(f"  Max height error: {np.max(np.abs(h_test_np - h_exact_test)):.6f}")
print(f"  Max velocity error: {np.max(np.abs(u_test_np - u_exact_test)):.6f}")

print(f"\nSolution images saved in '{output_dir}'")
torch.save(model.state_dict(), os.path.join(output_dir, 'improved_swe_pinn_model.pth'))
print("Improved model saved successfully!")

print("\n" + "="*60)
print("IMPROVED SHALLOW WATER EQUATIONS PINN SUMMARY")
print("="*60)
print(f"Training parameters:")
print(f"  Epochs: {epochs}")
print(f"  Initial condition weight: {lambda_ic}")
print(f"  Collocation points: {num_collocation_points}")
print(f"  Learning rate: {learning_rate}")
print(f"  Training time: {elapsed_time:.2f} seconds")
print(f"  Final total loss: {loss.item():.4e}")
print(f"  Final IC loss: {loss_initial.item():.4e}")
print(f"  Final PDE loss: {loss_pde.item():.4e}")
print("="*60)
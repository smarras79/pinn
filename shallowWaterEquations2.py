import torch 
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Parameters for Shallow Water Equations
g = 9.81        # Gravitational acceleration (m/s^2)
H0 = 0.25      # Mean water depth (m)

# Domain parameters
x_min, x_max = -np.pi, np.pi    # Spatial domain [m]
t_min, t_max = 0.0, 1.0         # Time domain [s]
L = x_max - x_min

# Loss function weights - Better balanced
lambda_ic = 500.0   # Reduced from 1000
lambda_pde = 10.0 # 10 gave better results
lambda_bc = 10.0

momentum_weight = 0.5

# Training parameters
num_initial_points = 1000
num_boundary_points = 200
epochs = 1200
num_collocation_points = 3000
learning_rate = 1e-3
num_time_steps = 20

# Initial condition parameters
dam_position = 0.0
h_left = 0.25
h_right = 0.0
u_left = 0.0
u_right = 0.0

num_frequencies=6

# Output directory
output_dir = "swe_solution_fixed_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

def fourier_features(x, t, num_frequencies=6):
    features = [x, t]
    for i in range(num_frequencies):
        for fn in [torch.sin, torch.cos]:
            features.append(fn(2**i * np.pi * x))
            features.append(fn(2**i * np.pi * t))
    return torch.cat(features, dim=1)
    
class ImprovedPINN_SWE(nn.Module):
    """
    Improved Physics-Informed Neural Network for 1D Shallow Water Equations
    """
    def __init__(self):
        super(ImprovedPINN_SWE, self).__init__()
        num_frequencies = 6
        input_dim = 2 + 4 * num_frequencies
        
        # Shared backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Linear(2 + 4 * num_frequencies, 128),  # updated input size
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        
        # Separate heads for h and u
        self.h_head = nn.Sequential(
            nn.Linear(128, 128),
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
        
        self.u_head = nn.Sequential(
            nn.Linear(128, 128),
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
        
        # Initialize weights
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, t):
        # Normalize inputs
        '''x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        t_norm = 2 * t / t_max - 1
        
        inputs = torch.cat([x_norm, t_norm], dim=1)'''
        inputs = fourier_features(x, t)
        features = self.backbone(inputs)
        
        # Raw outputs
        h_raw = self.h_head(features)
        u_raw = self.u_head(features)

        # Apply soft constraints for initial conditions
        # Use a smooth transition function instead of hard constraints
        #sigma = 10.0  # Controls sharpness of transition
        #ic_weight = torch.exp(-t / 0.01)
        
        # Initial conditions
        '''h_ic = torch.where(x < dam_position, 
                          torch.tensor(h_left, dtype=x.dtype, device=x.device), 
                          torch.tensor(0.0, dtype=x.dtype, device=x.device))
        u_ic = torch.zeros_like(u_raw)
        
        # Blend with smooth transition
        h = ic_weight * h_ic + (1 - ic_weight) * torch.relu(h_raw)
        u = ic_weight * u_ic + (1 - ic_weight) * u_raw
        
        return h, u '''
        return torch.relu(h_raw), u_raw
        #epsilon = 1e-3
        #return torch.clamp(h_raw, min=epsilon), u_raw


# Instantiate the network
model = ImprovedPINN_SWE()

# Optimizer with scheduled learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

def improved_physics_loss(h, u, x, t):
    """
    Improved physics-informed loss with better wet/dry handling
    """
    # Compute gradients
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    # Continuity equation: ∂h/∂t + ∂(hu)/∂x = 0
    hu = h * u
    hu_x = torch.autograd.grad(hu, x, grad_outputs=torch.ones_like(hu), create_graph=True)[0]
    continuity_residual = h_t + hu_x
    
    # Momentum equation with improved wet/dry treatment
    wet_threshold = 1e-3
    wet_mask = torch.sigmoid((h - wet_threshold) * 1000)  # Smooth transition
    
    # Standard momentum equation: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0
    momentum_residual = u_t + u * u_x + g * h_x
    
    # Apply momentum equation only in wet regions
    momentum_loss = torch.mean(wet_mask * torch.abs(momentum_residual))
    #momentum_loss = torch.mean(wet_mask * momentum_residual**2) #Disregard. Makes results worse with g=9.81
    #pde_weight = 1 + 5 * t_collocation / t_max
    #momentum_loss = torch.mean(pde_weight * wet_mask * torch.abs(momentum_residual)) #Disregard. Makes results worse with g=9.81

    
    # In dry regions, enforce u ≈ 0 and h ≈ 0
    dry_mask = 1 - wet_mask
    dry_h_loss = torch.mean(dry_mask * h**2)
    dry_u_loss = torch.mean(dry_mask * u**2)
    
    # Continuity equation loss
    #continuity_loss = torch.mean(continuity_residual**2)
    continuity_loss = torch.mean(torch.abs(continuity_residual))
    #continuity_loss = torch.mean(pde_weight* torch.abs(continuity_residual))
    
    total_pde_loss = continuity_loss + momentum_weight * momentum_loss + 0.1 * (dry_h_loss + dry_u_loss)
    
    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item(),
        'dry_h': dry_h_loss.item(),
        'dry_u': dry_u_loss.item()
    }

def initial_condition_loss(h_pred, u_pred, x_initial):
    """
    Improved initial condition loss
    """
    # True initial conditions
    h_true = torch.where(x_initial < dam_position, 
                        torch.tensor(h_left, dtype=h_pred.dtype, device=h_pred.device), 
                        torch.tensor(0.0, dtype=h_pred.dtype, device=h_pred.device))
    u_true = torch.zeros_like(u_pred)
    
    # Standard L2 loss
    loss_h = torch.mean((h_pred - h_true)**2)
    loss_u = torch.mean((u_pred - u_true)**2)
    
    return loss_h + loss_u

def boundary_condition_loss(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    """
    Simple outflow boundary conditions
    """
    # Minimize reflection by penalizing large velocities at boundaries
    return 0.01 * (torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2))

def exact_dam_break_solution(x, t):
    """
    Ritter's analytical solution for dam break
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(t, torch.Tensor):
        t = t.item()
    
    if t <= 1e-8:
        h_exact = np.where(x < dam_position, h_left, 0.0)
        u_exact = np.zeros_like(x)
        return h_exact, u_exact
    
    c0 = np.sqrt(g * h_left)
    x_left = dam_position - c0 * t
    x_right = dam_position + 2 * c0 * t
    
    h_exact = np.zeros_like(x)
    u_exact = np.zeros_like(x)
    
    # Left region (undisturbed)
    mask_left = x <= x_left
    h_exact[mask_left] = h_left
    u_exact[mask_left] = 0.0
    
    # Rarefaction fan
    mask_fan = (x > x_left) & (x < x_right)
    if np.any(mask_fan):
        xi = (x[mask_fan] - dam_position) / t
        h_exact[mask_fan] = (1.0 / (9.0 * g)) * (2 * c0 - xi)**2
        u_exact[mask_fan] = (2.0 / 3.0) * (xi + c0)
        
    # Right region (dry)
    mask_right = x >= x_right
    h_exact[mask_right] = 0.0
    u_exact[mask_right] = 0.0
    
    return h_exact, u_exact

# Generate training data
# More focused sampling near dam
# More focused sampling near dam
x_dam_dense = torch.linspace(dam_position - 0.5, dam_position + 0.5, num_collocation_points // 2).reshape(-1, 1)
x_outer = torch.rand(num_collocation_points // 2, 1) * (x_max - x_min) + x_min
x_collocation = torch.cat([x_dam_dense, x_outer], dim=0)

t_early = torch.rand(num_collocation_points // 2, 1) * 0.1
t_late = torch.rand(num_collocation_points // 2, 1) * 0.9 + 0.1 
t_collocation = torch.cat([t_early, t_late], dim=0)

# Initial condition points
x_initial = torch.linspace(x_min, x_max, num_initial_points).reshape(-1, 1)
t_initial = torch.zeros(num_initial_points, 1)

# Boundary points
x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
x_boundary_right = torch.ones(num_boundary_points, 1) * x_max
t_boundary = torch.rand(num_boundary_points, 1) * (t_max - t_min) + t_min

# Set requires_grad
x_collocation.requires_grad_(True)
t_collocation.requires_grad_(True)
x_boundary_left.requires_grad_(True)
x_boundary_right.requires_grad_(True)

print("Starting improved training for 1D Shallow Water Equations...")
print(f"Domain: x ∈ [{x_min}, {x_max}], t ∈ [{t_min}, {t_max}]")
print(f"Dam break at x = {dam_position}, h_left = {h_left}, h_right = {h_right}")

start_time = time.time()
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Initial condition loss
    h_initial_pred, u_initial_pred = model(x_initial, t_initial)
    loss_initial = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial)

    # Physics loss
    h_collocation, u_collocation = model(x_collocation, t_collocation)
    loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                    x_collocation, t_collocation)
    
    # Boundary loss
    h_boundary_left, u_boundary_left = model(x_boundary_left, t_boundary)
    h_boundary_right, u_boundary_right = model(x_boundary_right, t_boundary)
    loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                          h_boundary_right, u_boundary_right)

    # Total loss
    loss = lambda_ic * loss_initial + lambda_pde * loss_pde + lambda_bc * loss_boundary
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Total Loss: {loss.item():.4e}")
        print(f"  IC Loss: {loss_initial.item():.4e}")
        print(f"  PDE Loss: {loss_pde.item():.4e}")
        print(f"  Boundary Loss: {loss_boundary.item():.4e}")
        print(f"  Continuity: {pde_components['continuity']:.4e}")
        print(f"  Momentum: {pde_components['momentum']:.4e}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds.")

# Generate solution plots
x_plot = torch.linspace(x_min, x_max, 500).view(-1, 1)
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
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Water height
    ax1.plot(x_np, h_pred_plot, 'b-', label='PINN h(x,t)', linewidth=2)
    ax1.plot(x_np, h_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    ax1.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax1.set_ylabel('Water Height h(x,t) [m]')
    ax1.set_title(f'Water Height - t = {t_np:.3f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, h_left * 1.1)
    
    # Velocity
    ax2.plot(x_np, u_pred_plot, 'g-', label='PINN u(x,t)', linewidth=2)
    ax2.plot(x_np, u_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
    ax2.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
    ax2.set_ylabel('Velocity u(x,t) [m/s]')
    ax2.set_title(f'Velocity - t = {t_np:.3f}s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Height error
    ax3.semilogy(x_np, np.maximum(h_error, 1e-10), 'r-', linewidth=2)
    ax3.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax3.set_ylabel('|h_pred - h_exact|')
    ax3.set_title(f'Height Error - Max: {np.max(h_error):.4f}')
    ax3.grid(True, alpha=0.3)
    
    # Velocity error
    ax4.semilogy(x_np, np.maximum(u_error, 1e-10), 'g-', linewidth=2)
    ax4.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Position x [m]')
    ax4.set_ylabel('|u_pred - u_exact|')
    ax4.set_title(f'Velocity Error - Max: {np.max(u_error):.4f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add information text
    info_text = f"Epochs: {epochs} | IC Weight: {lambda_ic} | Points: {num_collocation_points}\n"
    info_text += f"Training time: {elapsed_time:.1f}s | Final loss: {loss.item():.4e}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9)
    
    plt.savefig(os.path.join(output_dir, f"swe_solution_t_{i:03d}.png"), dpi=150, bbox_inches='tight')
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

# Test initial condition accuracy
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

print(f"\nSolution images saved in '{output_dir}'")
torch.save(model.state_dict(), os.path.join(output_dir, 'improved_swe_pinn_model.pth'))
print("Model saved successfully!")
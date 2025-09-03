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
#t_min, t_max = 0.0, 5.0     # Time domain [s]
L = x_max - x_min

# Training parameters
num_initial_points = 1500 # INCREASED from 1000 for better IC sampling
num_boundary_points = 200
epochs = 1000
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

lambda_c = 1.0
lambda_m = 10.0  # Increase if momentum is underfitting

# Output directory
output_dir = "swe/temp/swe_solution_oneInput_case6_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

def get_weights(epoch, total_epochs):
    ic_weight = 100.0 #100
    pde_weight = 50.0 #10
    bc_weight = 10.0
    return ic_weight, pde_weight, bc_weight

# ------------------ Input Normalization ------------------
def normalize(x, xmin=0.0, xmax=20.0):
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

# def normalize_t(t, tmin=0.0, tmax=1.0):
#     return 2.0 * (t - tmin) / (tmax - tmin) - 1.0

class ImprovedPINN_SWE(nn.Module):
    """
    Improved Physics-Informed Neural Network for 1D Shallow Water Equations
    """
    def __init__(self):
        super(ImprovedPINN_SWE, self).__init__()
        #num_frequencies = 6
        input_dim = 2
        
        # Shared backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Linear(1, 128),  # updated input size
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

    def forward(self, x):
        # Normalize inputs: Neural networks train better with inputs in the range [-1, 1]
        x_norm = normalize(x)
        #t_norm = normalize_t(t)
        inputs = torch.cat([x_norm], dim=1)
        #inputs = torch.cat([x, t], dim=1)
        features = self.backbone(inputs)

        h_raw = self.h_head(features)
        u_raw = self.u_head(features)

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
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # Steady continuity residual: ∂(hu)/∂x = 0
    continuity_residual = hu_x

    # Momentum residual: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0 (only in wet regions)
    # Change the momentum residual to include the bed slope ∂zb/∂x
    epsilon = 1e-6 #Avoids large or exploding gradients when h → 0 (common near wet-dry interfaces or sharp dam fronts)
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb), create_graph=True)[0]
    momentum_residual = flux_x + g * h * dzb_dx

    # Wet/dry mask
    #wet_threshold = 0.02 #1e-1 # change these
    # Sigmoid is not giving good results. 
    #wet_mask = torch.sigmoid((h - wet_threshold) * 100)  # Smooth transition around wet/dry threshold
    wet_mask = (h > 1e-6).float() #0
    dry_mask = 1.0 - wet_mask
    dry_momentum_penalty = torch.mean(dry_mask * (h * u)**2)

    # Discontinuity detector: total gradient magnitude
    grad_strength = torch.abs(h_x) + torch.abs(u_x)

    # Gradient-based weighting: reduce loss impact where solution is steep
    weight_map = 1.0 / (1.0 + 0.5 * grad_strength.detach()) #best results with multiplier value of 0.5. Keep it in this range [0.1, 1.0]

    # Weighted PDE residuals
    continuity_loss = torch.mean(continuity_residual**2)
    momentum_loss = torch.mean(momentum_residual**2)
    #momentum_loss = torch.mean(wet_mask * momentum_residual**2)

    # Penalize dry regions gently
    dry_h_loss = torch.mean(dry_mask * h**2)
    dry_u_loss = torch.mean(dry_mask * u**2)

    # Dynamic weighting
    total_pde_grad = continuity_loss.item() + momentum_loss.item()
    if total_pde_grad > 0:
        lambda_c = momentum_loss.item() / total_pde_grad
        lambda_m = continuity_loss.item() / total_pde_grad

    # Combine total PDE loss
    total_pde_loss = continuity_loss + momentum_loss 
    #+ 0.1 * (dry_h_loss + dry_u_loss) 
    #+ 1.0 * dry_momentum_penalty 

    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item()
        #'dry_h': dry_h_loss.item(),
        #'dry_u': dry_u_loss.item()
    }

# ------------------ Bed Elevation Function ------------------

def bed_elevation(x: torch.Tensor) -> torch.Tensor:
    zb = torch.zeros_like(x)
    zb_h = 0.2 - 0.05 * (x - 10.0) **2
    return torch.where((x > 8.0) & (x < 12.0), zb_h, zb)

def initial_condition(x, case=6):
    if case == 6:
        eta_val, q_val = 0.33, 0.18
    elif case == 7:
        eta_val, q_val = 2.0, 4.42
    else:
        raise ValueError("Invalid case")
    
    zb = bed_elevation(x)
    eta = torch.ones_like(x) * eta_val
    h = eta - zb # water depth
    q = torch.ones_like(x) * q_val
    u = q / h
    return h, u

def initial_condition_loss(h_pred, u_pred, x, case=6):
    h_true, u_true = initial_condition(x, case)
    loss_h = torch.mean((h_pred - h_true) **2)
    loss_u = torch.mean((u_pred - u_true) **2)
    return loss_h + loss_u

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
    q = torch.tensor([[x_min]]) * q_val
    u = q / h
    return u
    #return torch.tensor([[0.0]])

def u_bc_right():
    h = eta_right()
    q = torch.tensor([[x_max]]) * q_val
    u = q / h
    return u
    #return torch.tensor([[0.0]])

# def u_bc():
#     return torch.tensor([[0.0]], dtype=torch.float32)
    #return torch.where(t < 0.0, torch.tensor(u_left), torch.tensor(u_right))  # Dam at x = 0.0

def boundary_condition_loss(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    """
    Simple outflow boundary conditions
    """

    loss_bc_left = torch.mean((h_left_pred - h_bc_left())**2) + torch.mean((u_left_pred - u_bc_left())**2)
    loss_bc_right = torch.mean((h_right_pred - h_bc_right())**2) + torch.mean((u_right_pred - u_bc_right())**2)
    return loss_bc_left + loss_bc_right
    #return 0.01 * (torch.mean(h_left_pred**2) + torch.mean(h_right_pred**2)) + 0.01 * (torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2))

# Generate training data
# Initial condition points
x_initial = torch.linspace(x_min, x_max, num_initial_points).reshape(-1, 1)

# Collocation points: More focused sampling near dam and early times
c0 = np.sqrt(g * eta_val)

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

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Initial condition loss
    h_initial_pred, u_initial_pred = model(x_initial)
    loss_initial = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial, case=6)

    # Physics loss
    h_collocation, u_collocation = model(x_collocation)
    loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                    x_collocation)
    
    # Boundary loss
    h_boundary_left, u_boundary_left = model(x_boundary_left)
    h_boundary_right, u_boundary_right = model(x_boundary_right)
    loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                          h_boundary_right, u_boundary_right)

    lambda_ic_curr, lambda_pde_curr, lambda_bc_curr = get_weights(epoch, epochs)

    # Total loss
    loss = (
        lambda_ic_curr * loss_initial
        + lambda_pde_curr * loss_pde
        + lambda_bc_curr * loss_boundary
    )

    loss.backward()
    
    # with torch.no_grad():
    #     h_train_mean = h_collocation.mean().item()
    #     h_train_std = h_collocation.std().item()
    #     h_train_max = h_collocation.max().item()
        #print(f"[Epoch {epoch+1}] h_mean: {h_train_mean:.4f}, h_std: {h_train_std:.4f}, h_max: {h_train_max:.4f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        #print(f"  Curriculum Weights - IC: {lambda_ic_curr:.1f}, PDE: {lambda_pde_curr:.1f}, BC: {lambda_bc_curr:.1f}")
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
# time_steps = torch.linspace(t_min, t_max, num_time_steps)

print("\nDiagnostic check: did the model learn anything...")
# Diagnostic check: did the model learn anything?
# t_plot = torch.ones_like(x_plot) * 0.2  # Choose any t > 0

with torch.no_grad():
    h_pred, u_pred = model(x_plot)
    print("Mean h:", h_pred.mean().item(), "Std h:", h_pred.std().item())
    print("Mean u:", u_pred.mean().item(), "Std u:", u_pred.std().item())

print("\nChecking initial condition prediction...")
model.eval()
x_test = torch.linspace(x_min, x_max, 500).view(-1, 1)
#t_test = torch.zeros_like(x_test)
with torch.no_grad():
    h_pred, u_pred = model(x_test)
    print("IC Check: Mean h:", h_pred.mean().item(), "Std h:", h_pred.std().item())
    print("IC Check: Mean u:", u_pred.mean().item(), "Std u:", u_pred.std().item())


print("\nChecking model for time steps...")
# for i, t_val in enumerate(time_steps):
#     t_plot = torch.ones_like(x_plot) * t_val
    
with torch.no_grad():
    h_pred, u_pred = model(x_plot)
    h_pred_plot = h_pred.numpy().flatten()
    u_pred_plot = u_pred.numpy().flatten()
    # Compute bed elevation and free surface
    zb_plot = bed_elevation(x_plot)
    eta_plot = h_pred_plot + zb_plot.numpy().flatten() # Free surface
    # STEP 3: Check PDE Residuals at this time step
    # Temporarily enable autograd
    #x_plot.requires_grad_(True)
    #t_plot.requires_grad_(True)

    # model.train()  # Needed for autograd to work properly
    # h_train, u_train = model(x_plot, t_plot)

    # # Compute physics residuals
    # pde_loss, residuals = improved_physics_loss(h_train, u_train, x_plot, t_plot)

    # print(f"[t = {t_val.item():.3f} s] PDE Loss = {pde_loss.item():.4e}")
    # print(f"  Continuity Residual: {residuals['continuity']:.4e}")
    # print(f"  Momentum Residual:   {residuals['momentum']:.4e}")
    # print(f"  Dry h Residual:      {residuals['dry_h']:.4e}")
    # print(f"  Dry u Residual:      {residuals['dry_u']:.4e}")
    # print("-" * 40)

    # model.eval()  # Go back to eval mode for plotting

    x_np = x_plot.detach().numpy().flatten()
    #t_np = t_val.item()
    
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

# Test initial condition accuracy
# print("\n" + "="*60)
# print("INITIAL CONDITION VALIDATION")
# print("="*60)

# x_test = torch.linspace(x_min, x_max, 100).view(-1, 1)
# t_test = torch.zeros_like(x_test)

# with torch.no_grad():
#     h_test, u_test = model(x_test, t_test)
#     h_test_np = h_test.numpy().flatten()
#     u_test_np = u_test.numpy().flatten()

# x_test_np = x_test.numpy().flatten()
# h_exact_test, u_exact_test = exact_dam_break_solution(x_test_np, 0.0)

# h_ic_error = np.mean(np.abs(h_test_np - h_exact_test))
# u_ic_error = np.mean(np.abs(u_test_np - u_exact_test))

# print(f"Initial condition errors:")
# print(f"  Water height MAE: {h_ic_error:.6f}")
# print(f"  Velocity MAE: {u_ic_error:.6f}")

print(f"\nSolution images saved in '{output_dir}'")
torch.save(model.state_dict(), os.path.join(output_dir, 'curriculum_swe_pinn_model.pth'))
print("Model saved successfully!")
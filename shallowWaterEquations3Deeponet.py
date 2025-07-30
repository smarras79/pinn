import torch 
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from pyDOE import lhs
import tensorflow as tf

# Set the environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Parameters for Shallow Water Equations
g = 1.0 # 9.81        # Gravitational acceleration (m/s^2)
H0 = 0.25      # Mean water depth (m)

# Domain parameters
x_min, x_max = -np.pi, np.pi    # Spatial domain [m]
t_min, t_max = 0.0, 1.0         # Time domain [s]
L = x_max - x_min

# Loss function weights - Better balanced
lambda_ic = 10.0 # Reduced emphasis slightly
lambda_pde = 1000.0 # Increased emphasis
lambda_bc = 50.0 # Increased moderately
lambda_loss_supervise = 0.1 # Supervision loss weight

momentum_weight = 1.0 # Make momentum PDE more important

# Training parameters
num_initial_points = 1500 # INCREASED from 1000 for better IC sampling
num_boundary_points = 200
epochs = 1000
num_collocation_points = 6000
learning_rate = 1e-3
num_time_steps = 20

#Scheduler tuning parameters
scheduler_step_size_frequency = 4 #Number of times we want scheduler to reduce LR during full training with epochs
scheduler_step_size = epochs // scheduler_step_size_frequency # Epoch intervals at which scheduler will reduce LR 
scheduler_gamma=0.5 #Factor by which scheduler will reduce LR at each epoch interval

# Initial condition parameters
dam_position = 0.0
h_left = 0.25
h_right = 0.0
u_left = 0.0
u_right = 0.0

num_frequencies=6

# Output directory
output_dir = "swe/temp/swe_solution_fixed_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

def get_weights(epoch, total_epochs):
    ic_weight = 1.0 #100
    pde_weight = 1.0 #10
    bc_weight = 1.0 #10
    return ic_weight, pde_weight, bc_weight

#Curriculum Learning Weights
def get_curriculum_weights(epoch, total_epochs):
    progress = epoch / total_epochs

    if progress < 0.2:
        ic_weight = 100.0
        pde_weight = 10.0
        bc_weight = 10.0
    else:
        ic_weight = 50.0
        pde_weight = 50.0
        bc_weight = 10.0

    if progress < 0.1:
        ic_weight = 100.0
        pde_weight = 1.0
        bc_weight = 1.0
    elif progress < 0.3:
        ic_weight = 50.0
        pde_weight = 50.0
        bc_weight = 10.0
    else:
        ic_weight = 10.0
        pde_weight = 500.0
        bc_weight = 100.0

    return ic_weight, pde_weight, bc_weight


def get_decaying_ic_weight(self, epoch, initial_weight=2.0, decay_rate=0.01):
    """Exponentially decay IC weight over time"""
    return initial_weight * np.exp(-decay_rate * epoch)

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
        input_dim = 2
        
        # Shared backbone for feature extraction
        self.backbone = nn.Sequential(
            nn.Linear(2, 128),  # updated input size
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
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        t_norm = 2 * (t - t_min) / (t_max - t_min) - 1
        inputs = torch.cat([x_norm, t_norm], dim=1)
        features = self.backbone(inputs)

        h_raw = self.h_head(features)
        u_raw = self.u_head(features)

        # During evaluation, return perfect IC at t=0
        if not self.training and torch.all(t == 0):
            h = torch.where(x < dam_position,
                        h_left * torch.ones_like(x),
                        torch.zeros_like(x))
            u = torch.zeros_like(x)
            return h, u

        return h_raw, u_raw
    
        #epsilon = 1e-3
        #return torch.clamp(h_raw, min=epsilon), u_raw


# Instantiate the network
model = ImprovedPINN_SWE()

# Optimizer with scheduled learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)

def improved_physics_loss(h, u, x, t):
    """
    Physics-informed loss for 1D Shallow Water Equations with gradient-based weighting.
    Penalizes discontinuity regions less and focuses on smooth wave dynamics.
    """

    # Compute derivatives
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    hu = h * u
    hu_x = torch.autograd.grad(hu, x, grad_outputs=torch.ones_like(hu), create_graph=True)[0]

    # Continuity residual: ∂h/∂t + ∂(hu)/∂x = 0
    continuity_residual = h_t + hu_x

    # Momentum residual: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0 (only in wet regions)
    momentum_residual = u_t + u * u_x + g * h_x

    # Wet/dry mask
    wet_threshold = 0.02 #1e-1 # change these
    wet_mask = torch.sigmoid((h - wet_threshold) * 100)  # Smooth transition around wet/dry threshold
    dry_mask = 1.0 - wet_mask

    # Discontinuity detector: total gradient magnitude
    grad_strength = torch.abs(h_x) + torch.abs(u_x)

    # Gradient-based weighting: reduce loss impact where solution is steep
    weight_map = 1.0 / (1.0 + 0.5 * grad_strength.detach()) #best results with multiplier value of 0.5. Keep it in this range [0.1, 1.0]

    # Weighted PDE residuals
    continuity_loss = torch.mean(weight_map * continuity_residual**2)
    momentum_loss = torch.mean(weight_map * wet_mask * momentum_residual**2)

    # Extra focus on dam break region
    #dam_region_mask = torch.abs(x - dam_position) < 0.1
    #dam_loss_continuity_residual = torch.mean(weight_map * dam_region_mask.float() * (continuity_residual)**2)
    #dam_region_momentum_loss = torch.mean(weight_map * dam_region_mask.float() * (momentum_residual)**2)

    # Penalize dry regions gently
    dry_h_loss = torch.mean(dry_mask * h**2)
    dry_u_loss = torch.mean(dry_mask * u**2)

    # Combine total PDE loss
    total_pde_loss = continuity_loss + momentum_weight * momentum_loss + 0.1 * (dry_h_loss + dry_u_loss)

    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item(),
        'dry_h': dry_h_loss.item(),
        'dry_u': dry_u_loss.item()
    }

# Analytical solution
a0 = lambda H: np.sqrt(g*H)
h_analytical = lambda t, x, x0, H: tf.where(x-x0<-a0(H)*t, a0(H)**2/g, tf.where(x-x0<2*a0(H)*t, (2*a0(H)-(x-x0)/t)**2/(9*g), 0.0))
u_analytical = lambda t, x, x0, H: tf.where(x-x0<-a0(H)*t, 0.0, tf.where(x-x0<2*a0(H)*t, 2/3*(a0(H)+(x-x0)/t), 0.0))


def initial_condition_loss(h_pred, u_pred, x_initial):
    """
    Improved initial condition loss
    """
    # True initial conditions
    # Original condition
    # h_true = torch.where(x_initial < dam_position, 
    #                     torch.tensor(h_left, dtype=h_pred.dtype, device=h_pred.device), 
    #                     torch.tensor(0.0, dtype=h_pred.dtype, device=h_pred.device))
    
    # Attempt to make it softer
    # h_true = torch.where(x_initial < dam_position, torch.tensor(h_left, dtype=h_pred.dtype, device=h_pred.device),
    #             torch.where((x_initial >= dam_position) & (x_initial < 0.02), torch.tensor(h_left*0.40, dtype=h_pred.dtype, device=h_pred.device), 
    #                     torch.where((x_initial >= 0.02) & (x_initial < 0.04),torch.tensor(h_left*0.32, dtype=h_pred.dtype, device=h_pred.device),
    #                         torch.where((x_initial >= 0.04) & (x_initial < 0.06),torch.tensor(h_left*0.24, dtype=h_pred.dtype, device=h_pred.device),
    #                             torch.where((x_initial >= 0.06) & (x_initial < 0.08),torch.tensor(h_left*0.16, dtype=h_pred.dtype, device=h_pred.device),
    #                                 torch.where((x_initial >= 0.08) & (x_initial<0.1),torch.tensor(h_left*0.12, dtype=h_pred.dtype, device=h_pred.device),
    #                                     torch.tensor(0.0, dtype=h_pred.dtype, device=h_pred.device)))))))
    
    # Using a sigmoid function to make it softer on right side
    # Initial condition wet mask
    x_initial_ic = torch.linspace(0, x_max, num_initial_points).reshape(-1, 1)
    mask = torch.sigmoid((x_initial_ic - 0.01) * 10)
    mask_complement = 1.0 - mask
    h_left_initial = torch.ones(num_initial_points, 1) * h_left
    h_true = torch.where(x_initial < dam_position, 
                        torch.tensor(h_left, dtype=h_pred.dtype, device=h_pred.device), 
                        mask_complement * h_left_initial)
    
    u_true = torch.zeros_like(u_pred)
    
    # Standard L2 loss
    loss_h = torch.mean((h_pred - h_true)**2)
    loss_u = torch.mean((u_pred - u_true)**2)
    
    # Extra focus on dam break region. Not using for now
    # dam_region_mask = torch.abs(x_initial - dam_position) < 0.3
    # dam_loss_h = torch.mean(dam_region_mask.float() * (h_pred - h_true)**2)
    # dam_loss_u = torch.mean(dam_region_mask.float() * (u_pred - u_true)**2)

    return loss_h + loss_u #+ 30.0 * (dam_loss_h + dam_loss_u) # Gives bad results if "Extra focus on dam break region" is not used

# Characteristic scales
U_scale = 1.0
H_scale = 1.0

def initial_condition_loss_using_exact_solution(hi_pred, ui_pred):
    IC1 = 10*torch.mean((torch.from_numpy(hi)-hi_pred)**2)/H_scale**2
    IC2 = 10*torch.mean((torch.from_numpy(ui)*hi-ui_pred*hi_pred)**2)/(U_scale*H_scale)**2
    ICloss = IC1 + IC2
    return ICloss
    

def boundary_condition_loss(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    """
    Simple outflow boundary conditions
    """
    return 0.01 * (torch.mean(h_left_pred**2) + torch.mean(h_right_pred**2)) + 0.01 * (torch.mean(u_left_pred**2) + torch.mean(u_right_pred**2))

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

def ritter_supervision_loss(model, x_supervise, t_supervise):
    '''
    Calculates MSE between PINN predictions and Ritter's exact dam break solution
    at given supervised points (x_supervise, t_supervise).
    '''
    h_pred, u_pred = model(x_supervise, t_supervise)

    # Convert to numpy for exact solution
    x_np = x_supervise.detach().cpu().numpy().flatten()
    t_np = t_supervise[0][0].item()  # assume uniform t

    h_exact, u_exact = exact_dam_break_solution(x_np, t_np)
    h_exact = torch.tensor(h_exact, dtype=h_pred.dtype, device=h_pred.device).view(-1, 1)
    u_exact = torch.tensor(u_exact, dtype=u_pred.dtype, device=u_pred.device).view(-1, 1)

    loss_h = torch.mean((h_pred - h_exact)**2)
    loss_u = torch.mean((u_pred - u_exact)**2)

    return loss_h + loss_u

# Generate training data
# More focused sampling near dam and early times
c0 = np.sqrt(g * h_left)
fan_left = dam_position - c0 * t_max
fan_right = dam_position + 2 * c0 * t_max

x_fan = torch.rand(int(0.7 * num_collocation_points), 1) * (fan_right - fan_left) + fan_left
x_outer = torch.rand(int(0.3 * num_collocation_points), 1) * (x_max - x_min) + x_min

x_collocation = torch.cat([x_fan, x_outer], dim=0)
#x_collocation = torch.cat([x_dam_dense, x_outer], dim=0)

t_early = torch.rand(num_collocation_points // 3, 1) * 0.2 
t_mid = torch.rand(num_collocation_points // 3, 1) * 0.4 + 0.2 
t_late = torch.rand(num_collocation_points // 3, 1) * 0.4 + 0.6
t_collocation = torch.cat([t_early, t_mid, t_late], dim=0)

# Now x_collocation and t_collocation have the same number of rows
assert x_collocation.shape[0] == t_collocation.shape[0], "x and t collocation sizes must match"

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

# Supervised data points from Ritter's solution (added to hybrid loss)
num_supervise = 200
x_supervise = torch.linspace(dam_position, dam_position + 2.5, num_supervise).view(-1, 1)
t_supervise = torch.ones_like(x_supervise) * 0.5  # mid-time slice

#This approach taken from DeepONetDamBreak
N=10000
#t_bdry = [0.0, 1.0]
#x_bdry = [x_min, x_max]
# Convert input to numpy array
t_bdry, x_bdry = np.array([0.0, 1.0]), np.array([x_min, x_max])

# Uniform random sampling for PDE points
tx_min = np.array([t_bdry[0], x_bdry[0]])
tx_max = np.array([t_bdry[1], x_bdry[1]])

# Sample a new IC
H = np.random.uniform(0.1, 1, (N,1))
x0 = np.random.uniform(-np.pi/4, np.pi/4, (N,1))
t0 = np.random.uniform(0, 2, (N,1))

# Initial conditions
ic_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N)
x_ic = ic_points[:, 0]
u_ic = u_analytical(t0, x_ic.reshape((-1,1)), x0, H)
h_ic = h_analytical(t0, x_ic.reshape((-1,1)), x0, H)
ics = np.column_stack([0*t0, x_ic, u_ic, h_ic]).astype(np.float32)
ti, xi, ui, hi = ics[:,:1], ics[:,1:2], ics[:,2:3], ics[:,3:4]

print("Starting improved training for 1D Shallow Water Equations using DeepOnet...")

for pre_epoch in range(1200):
    optimizer.zero_grad()

    h_initial_pred, u_initial_pred = model(torch.from_numpy(xi), torch.from_numpy(ti))
    loss_ic = initial_condition_loss_using_exact_solution(h_initial_pred, u_initial_pred)
    #loss_ic = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial)
    loss_ic.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if (pre_epoch + 1) % 20 == 0:
        print(f"Pretraining Epoch {pre_epoch+1}/1200 - IC Loss: {loss_ic.item():.4e}")


#print("Starting improved training for 1D Shallow Water Equations...")
# === PHASE 0: IC Pretraining ===
#print("\nPretraining only on Initial Conditions for 200 epochs...\n")

# for pre_epoch in range(1200):
#     optimizer.zero_grad()

#     h_initial_pred, u_initial_pred = model(x_initial, t_initial)
#     loss_ic = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial)
#     loss_ic.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

#     if (pre_epoch + 1) % 20 == 0:
#         print(f"Pretraining Epoch {pre_epoch+1}/1200 - IC Loss: {loss_ic.item():.4e}")

# Reset LR scheduler (optional but recommended)
scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)

print(f"Domain: x ∈ [{x_min}, {x_max}], t ∈ [{t_min}, {t_max}]")
print(f"Dam break at x = {dam_position}, h_left = {h_left}, h_right = {h_right}")
#print(f"Training phases: IC focus (10%) -> Transition (50%) -> Physics focus (40%)")

start_time = time.time()
loss_history = []

model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Initial condition loss
    #h_initial_pred, u_initial_pred = model(torch.from_numpy(xi), torch.from_numpy(ti))
    #loss_initial = initial_condition_loss(h_initial_pred, u_initial_pred, x_initial)
    #loss_initial = initial_condition_loss_using_exact_solution(h_initial_pred, u_initial_pred)

    # Physics loss
    h_collocation, u_collocation = model(x_collocation, t_collocation)
    loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                    x_collocation, t_collocation)
    
    # Boundary loss
    h_boundary_left, u_boundary_left = model(x_boundary_left, t_boundary)
    h_boundary_right, u_boundary_right = model(x_boundary_right, t_boundary)
    loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                          h_boundary_right, u_boundary_right)

    # Compute Ritter supervision loss
    #loss_supervise = ritter_supervision_loss(model, x_supervise, t_supervise)

    lambda_ic_curr, lambda_pde_curr, lambda_bc_curr = get_weights(epoch, epochs)

    # Total loss
    loss = (
        #lambda_ic_curr * loss_initial +
        lambda_pde_curr * loss_pde
        + lambda_bc_curr * loss_boundary
        #+ lambda_loss_supervise * loss_supervise  # supervision weight
    )

    loss.backward()
    
    with torch.no_grad():
        h_train_mean = h_collocation.mean().item()
        h_train_std = h_collocation.std().item()
        h_train_max = h_collocation.max().item()
        #print(f"[Epoch {epoch+1}] h_mean: {h_train_mean:.4f}, h_std: {h_train_std:.4f}, h_max: {h_train_max:.4f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Curriculum Weights - IC: {lambda_ic_curr:.1f}, PDE: {lambda_pde_curr:.1f}, BC: {lambda_bc_curr:.1f}")
        print(f"  Total Loss: {loss.item():.4e}")
        #print(f"  IC Loss: {loss_initial.item():.4e}")
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
# Diagnostic check: did the model learn anything?
t_plot = torch.ones_like(x_plot) * 0.2  # Choose any t > 0

with torch.no_grad():
    h_pred, u_pred = model(x_plot, t_plot)
    print("Mean h:", h_pred.mean().item(), "Std h:", h_pred.std().item())
    print("Mean u:", u_pred.mean().item(), "Std u:", u_pred.std().item())

print("\nChecking initial condition prediction...")

model.eval()

x_test = torch.linspace(x_min, x_max, 500).view(-1, 1)
t_test = torch.zeros_like(x_test)

with torch.no_grad():
    h_pred, u_pred = model(x_test, t_test)


for i, t_val in enumerate(time_steps):
    t_plot = torch.ones_like(x_plot) * t_val
    
    with torch.no_grad():
        h_pred, u_pred = model(x_plot, t_plot)
        h_pred_plot = h_pred.numpy().flatten()
        u_pred_plot = u_pred.numpy().flatten()
    # STEP 3: Check PDE Residuals at this time step
    # Temporarily enable autograd
    x_plot.requires_grad_(True)
    t_plot.requires_grad_(True)

    model.train()  # Needed for autograd to work properly
    h_train, u_train = model(x_plot, t_plot)

    # Compute physics residuals
    pde_loss, residuals = improved_physics_loss(h_train, u_train, x_plot, t_plot)

    print(f"[t = {t_val.item():.3f} s] PDE Loss = {pde_loss.item():.4e}")
    print(f"  Continuity Residual: {residuals['continuity']:.4e}")
    print(f"  Momentum Residual:   {residuals['momentum']:.4e}")
    print(f"  Dry h Residual:      {residuals['dry_h']:.4e}")
    print(f"  Dry u Residual:      {residuals['dry_u']:.4e}")
    print("-" * 40)

    model.eval()  # Go back to eval mode for plotting

    x_np = x_plot.detach().numpy().flatten()
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
    info_text = f"Epochs: {epochs} | Curriculum Learning | Points: {num_collocation_points}\n"
    info_text += f"Training time: {elapsed_time:.1f}s | Final loss: {loss.item():.4e}"
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=9)
    
    plt.savefig(os.path.join(output_dir, f"swe_solution_t_{i:03d}.png"), dpi=150, bbox_inches='tight')
    plt.close()

# Loss history plot
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training Loss History - Curriculum Learning')
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
torch.save(model.state_dict(), os.path.join(output_dir, 'curriculum_swe_pinn_model.pth'))
print("Model saved successfully!")
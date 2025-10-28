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
x_min, x_max = 0.0, 25.0    # Spatial domain [m]
L = x_max - x_min
# Training parameters
num_collocation_points = 6000
num_boundary_points = 500
epochs = 10000
learning_rate = 1e-3
num_time_steps = 20
#Scheduler tuning parameters
scheduler_step_size_frequency = 5 #Number of times we want scheduler to reduce LR during full training with epochs
scheduler_step_size = epochs // scheduler_step_size_frequency # Epoch intervals at which scheduler will reduce LR 
scheduler_gamma=0.5 #Factor by which scheduler will reduce LR at each epoch interval

#Bump Region
x_bump_left = 8.0
x_bump_center = 10.0
x_bump_right = 12.0
bump_height = 0.2
bump_width = 4

#Bump left region
x_before_bump_left = 0
x_before_bump_right = 8

gradient_based_weighting = True

# Output directory
output_dir = "swe/temp/swe_steady_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

dir = 'C:\\Users\\rhear\\MPAS\\PINN\\DamBreak' # HLLC Analytical solution dir path
dire = 'C:\\Users\\rhear\\MPAS\\PINN\\DamBreak\\swe\\hllc' # HLLC Analytical solution dir path

loss_history = []
pde_loss_history = []
momentum_loss_history = []
continuity_loss_history = []
q_loss_history = []
loss_constraint_history = []
loss_constraint_velocity_history = []
loss_constraint_height_before_bump_history = []
loss_constraint_height_after_hydraulic_jump_history = []
loss_constraint_velocity_after_hydraulic_jump_history = []
x_collocation_fr_critical_max_history = []


class ImprovedPINN_SWE(nn.Module):
    """
    Improved Physics-Informed Neural Network for 1D Shallow Water Equations
    """
    def __init__(self):
        super(ImprovedPINN_SWE, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(1, 128)  

        # Hidden layers
        self.hidden_layers = nn.ModuleList()  
        self.hidden_layers.append(nn.Linear(128, 128))    
        self.hidden_layers.append(nn.Linear(128, 64))
        self.hidden_layers.append(nn.Linear(64, 32))
        self.hidden_layers.append(nn.Linear(32, 16))

        # Output layers for the two results
        self.output_h = nn.Linear(16, 1)
        self.output_u = nn.Linear(16, 1)

        # Activation function
        self.activation = nn.Tanh()
        
        # Initialize weights
        self.apply(self._init_weights)

        self.loss_func = nn.MSELoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Pass through input and hidden layers
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Get the two outputs
        h_raw = self.output_h(x)
        u_raw = self.output_u(x)
        epsilon = 1e-3
        return torch.clamp(h_raw, min=epsilon), u_raw
    
    # Define the Constraint Loss ---
    def constraint_loss(self, x_bump, h_pred):
        """Penalty for h < zb at bump region."""
        zb_val = bed_elevation(x_bump)
        violation = zb_val - h_pred # The penalty is applied only when h_pred is less than zb_val
        penalty = torch.clamp(violation, min=0)     # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))

    def constraint_height_loss_before_bump(self, x_bump, h_pred):
        violation = h_bc_left() - h_pred # ideally h_pred should be more than h_bc_left (height goes up)
        penalty = torch.clamp(violation, min=0) # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))

    def constraint_velocity_loss_before_bump(self, x_bump, u_pred):
        violation = u_pred - u_bc_left() # ideally u_pred should be less than u_bc_left(velocity goes down)
        penalty = torch.clamp(violation, min=0) # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))
    
    def constraint_loss_velocity(self, u_zeroes, u_pred):
        violation = u_zeroes - u_pred
        penalty = torch.clamp(violation, min=0) # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))
    
    def constraint_height_loss_after_hydraulic_jump(self, h_pred):
        violation = eta_right() - h_pred 
        penalty = torch.clamp(violation, min=0)     # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))

    def constraint_height_loss_after_hydraulic_jump_high_side(self, h_pred):
        violation = h_pred - eta_right()
        penalty = torch.clamp(violation, min=0)     # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))

    def constraint_velocity_loss_after_hydraulic_jump(self,u_pred,h_pred):
        violation = u_pred - u_bc_right() # u_pred should not be higher than u_bc_right
        penalty = torch.clamp(violation, min=0)     # Use torch.clamp to penalize only positive violations
        return self.loss_func(penalty, torch.zeros_like(penalty))


# Instantiate the network
model = ImprovedPINN_SWE()

# Optimizer with scheduled learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)

def improved_physics_loss(h, u, x, epoch,epochs):
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
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    zb = bed_elevation(x)

    #friction slope
    # manning = 0.0
    # sfx = manning**2 * u * torch.abs(u) / h.clamp(min=1e-3)**(4.0/3.0)

    # Steady continuity residual: ∂(hu)/∂x = 0
    continuity_residual = hu_x

    # Momentum residual: ∂u/∂t + u∂u/∂x + g∂h/∂x = 0 (only in wet regions)
    # Change the momentum residual to include the bed slope ∂zb/∂x
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb), create_graph=True)[0]

    bed_momentum = g * h * dzb_dx
    momentum_residual = flux_x + bed_momentum #+ g * h * sfx

    # Discontinuity detector: total gradient magnitude
    if gradient_based_weighting == True:
        grad_strength = torch.abs(h_x) + torch.abs(u_x)
        # Gradient-based weighting: reduce loss impact where solution is steep
        weight_map = 1.0 / (1.0 + 0.8 * grad_strength.detach()) #best results with multiplier value of 0.5. Keep it in this range [0.1, 1.0]

    if gradient_based_weighting == True:
        # Weighted PDE residuals
        continuity_loss = torch.mean( weight_map * continuity_residual**2)
        momentum_loss = torch.mean( weight_map * momentum_residual**2)
    else:
        continuity_loss = torch.mean(continuity_residual**2)
        momentum_loss = torch.mean(momentum_residual**2)

    if epoch==epochs-1:
        logdata(x,h,u,q,hu_x,flux_x,bed_momentum)

    # Combine total PDE loss
    total_pde_loss = continuity_loss + momentum_loss 

    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item()
    }

# ------------------ Bed Elevation Function ------------------

def bed_elevation(x: torch.Tensor) -> torch.Tensor:
    zb = torch.zeros_like(x)
    zb_h = bump_height - (bump_height/bump_width) * (x - x_bump_center) **2
    return torch.where((x > x_bump_left) & (x < x_bump_right), zb_h, zb)

def set_eta_q(case=6):
    global eta_val,q_val
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

def boundary_condition_loss_left_side(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    loss_bc_left = torch.mean((h_left_pred - h_bc_left())**2) + torch.mean((u_left_pred - u_bc_left())**2)
    return  (loss_bc_left)

def boundary_condition_loss_right_side(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    loss_bc_right = torch.mean((h_right_pred - h_bc_right())**2) + torch.mean((u_right_pred - u_bc_right())**2)
    return  loss_bc_right

def boundary_condition_loss_both_side(h_left_pred, u_left_pred, h_right_pred, u_right_pred):
    loss_bc_left = torch.mean((h_left_pred - h_bc_left())**2) + torch.mean((u_left_pred - u_bc_left())**2)
    loss_bc_right = torch.mean((h_right_pred - h_bc_right())**2) + torch.mean((u_right_pred - u_bc_right())**2)
    return  (loss_bc_left +  loss_bc_right)

def train():
    # Generate training data. Enhanced training data sampling. Much denser sampling near bump
    x_bump_region = torch.linspace(x_bump_left, x_bump_right, num_collocation_points // 3)
    x_outer_left = torch.linspace(x_min, x_bump_left, num_collocation_points // 3)
    x_outer_right = torch.linspace(x_bump_right, x_max, num_collocation_points // 3)
    x_collocation = torch.cat([x_outer_left, x_bump_region, x_outer_right]).reshape(-1, 1)
    
    # Boundary points
    x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
    x_boundary_right = torch.ones(num_boundary_points, 1) * x_max

    # Bump region points
    x_bump_collocation = torch.cat([torch.linspace(x_bump_left, x_bump_right, num_collocation_points)]).reshape(-1, 1)

    # Bump before points
    x_before_bump_collocation = torch.cat([torch.linspace(x_before_bump_left, x_before_bump_right, num_collocation_points)]).reshape(-1, 1)

    u_zeroes = torch.zeros(num_collocation_points, 1)

    # Set requires_grad
    x_collocation.requires_grad_(True)
    x_boundary_left.requires_grad_(True)
    x_boundary_right.requires_grad_(True)
    x_bump_collocation.requires_grad_(True)
    x_before_bump_collocation.requires_grad_(True)

    print(f"Domain: x ∈ [{x_min}, {x_max}]")

    start_time = time.time()
    model.train()

    #Setting test case here. 6: supercritical. 7:subcritical
    #set_eta_q(test_case)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics loss
        h_collocation, u_collocation = model(x_collocation)

        epsilon = 1e-3
        Fr = u_collocation / torch.sqrt(g * torch.clamp(h_collocation, min=epsilon))

        loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                        x_collocation,epoch,epochs)
    
        # Boundary loss
        h_boundary_left, u_boundary_left = model(x_boundary_left)
        h_boundary_right, u_boundary_right = model(x_boundary_right)
        loss_boundary = boundary_condition_loss_left_side(h_boundary_left, u_boundary_left, 
                                            h_boundary_right, u_boundary_right)

        # Height contraint loss in "bump region". Height should not penetrate the bump
        h_collocation_bump, u_collocation_bump = model(x_bump_collocation)
        loss_constraint = model.constraint_loss(x_bump_collocation,h_collocation_bump)

        # Velocity contraint loss in full domain. velocity should not get negative.
        loss_constraint_velocity = model.constraint_loss_velocity(u_zeroes,u_collocation)

        # Height Constraint loss in "before bump region". Height should not be below eta_val
        h_collocation_before_bump, u_collocation_before_bump = model(x_before_bump_collocation)
        loss_constraint_height_before_bump = \
            model.constraint_height_loss_before_bump(x_before_bump_collocation,h_collocation_before_bump)

        # Velocity Constraint loss in "before bump region".
        loss_constraint_velocity_before_bump = \
            model.constraint_velocity_loss_before_bump(x_before_bump_collocation,u_collocation_before_bump)

        # Total loss
        pde_weight = 1
        bc_weight = 1
        loss_constraint_weight = 10 #In Bump: Height should not penetrate the bump
        loss_constraint_before_bump_weight = 1 #Before bump: Height should not be below eta_val
        loss_constraint_velocity_weight = 1 #Full domain: Velocity should not get negative
        loss = (
            pde_weight * loss_pde
            + bc_weight * loss_boundary
            + loss_constraint_weight * loss_constraint
            + loss_constraint_velocity_weight * loss_constraint_velocity
            + loss_constraint_before_bump_weight * loss_constraint_height_before_bump
            + loss_constraint_before_bump_weight * loss_constraint_velocity_before_bump
        )

        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        pde_loss_history.append(loss_pde.item())
        continuity_loss_history.append(pde_components['continuity'])
        momentum_loss_history.append(pde_components['momentum'])
        loss_constraint_history.append(loss_constraint.item())
        loss_constraint_velocity_history.append(loss_constraint_velocity.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {loss.item():.4e}")
            print(f"  PDE Loss: {loss_pde.item():.4e}")
            print(f"  Boundary Loss: {loss_boundary.item():.4e}")
            print(f"  Continuity: {pde_components['continuity']:.4e}")
            print(f"  Momentum: {pde_components['momentum']:.4e}")
            print(f"  loss_constraint: {loss_constraint.item():.4e}")
            print(f"  loss_constraint_velocity: {loss_constraint_velocity.item():.4e}")
            print(f"  Froude number: {torch.max(Fr):.4e}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    #End of Training loop

    # Training loop - Refinement
    print("Refinement training loop in bump region")
    # Bump region surrounding points
    x_around_bump_collocation = torch.cat([torch.linspace(x_bump_left-2, x_bump_right+2, num_collocation_points)]).reshape(-1, 1)
    x_around_bump_collocation.requires_grad_(True)
    epochsRefinement = 4000
    for epoch in range(epochsRefinement):
        optimizer.zero_grad()
        # Boundary loss
        h_boundary_left, u_boundary_left = model(x_boundary_left)
        h_boundary_right, u_boundary_right = model(x_boundary_right)
        loss_boundary = boundary_condition_loss_right_side(h_boundary_left, u_boundary_left, 
                                            h_boundary_right, u_boundary_right)

        h_around_collocation_bump, u_around_collocation_bump = model(x_around_bump_collocation)
        loss_pde, pde_components = improved_physics_loss(h_around_collocation_bump, u_around_collocation_bump, 
                                                        x_around_bump_collocation,epoch,epochsRefinement)

        # Height contraint loss in "bump region". Height should not penetrate the bump
        h_collocation_bump, u_collocation_bump = model(x_bump_collocation)
        loss_constraint = model.constraint_loss(x_bump_collocation,h_collocation_bump)

        # Hydraulic jump detection and height constraint
        epsilon = 1e-3
        Fr = u_collocation_bump / torch.sqrt(g * torch.clamp(h_collocation_bump, min=epsilon))

        lossrefinement = (
            loss_pde \
            + loss_boundary \
            + 5 * loss_constraint \
        )
        
        lossrefinement.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochsRefinement}")
            print(f"  Total Loss: {loss.item():.4e}")
            print(f"  Boundary Loss: {loss_boundary.item():.4e}")
            print(f"  PDE Loss: {loss_pde.item():.4e}")
            print(f"  Continuity: {pde_components['continuity']:.4e}")
            print(f"  Momentum: {pde_components['momentum']:.4e}")
            print(f"  Bump Height penetration Loss: {loss_constraint.item():.4e}")
            print(f"  Froude number: {torch.max(Fr):.4e}")
    #End of Training loop - Refinement

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    return loss


def test(loss,start_time):
    elapsed_time = time.time() - start_time
    # Generate solution plots
    #x_plot = torch.linspace(x_min, x_max, 500).view(-1, 1)
    
    #Using x_plot from analytical graphs x points
    x_plot = getxplot(test_case)

    #Analytical results
    h_exact = get_analytical_results(test_case)

    os.chdir(dir)

    print("\nDiagnostic check: did the model learn anything...")
    with torch.no_grad():
        h_pred, u_pred = model(x_plot)
        print("Mean h:", h_pred.mean().item(), "Std h:", h_pred.std().item())
        print("Mean u:", u_pred.mean().item(), "Std u:", u_pred.std().item())

    print("\nChecking model for time steps...")
    model.eval()

    with torch.no_grad():
        h_pred, u_pred = model(x_plot)
        h_pred_plot = h_pred.numpy().flatten()
        u_pred_plot = u_pred.numpy().flatten()
        # Compute bed elevation and free surface
        zb_plot = bed_elevation(x_plot)
        #eta_plot = h_pred_plot + zb_plot.numpy().flatten() # Free surface
        x_np = x_plot.detach().numpy().flatten()

        # Compure Froude number
        Fr = u_pred / torch.sqrt(g * h_pred)
        Fr_plot = Fr.numpy().flatten()
        fr_critical_mask = (Fr > 1.0).float()
        fr_critical_mask_plot = fr_critical_mask.numpy().flatten()

        # Calculate errors
        h_error = np.abs(h_pred_plot - h_exact)
        h_error_ratio = np.abs(h_pred_plot - h_exact)/np.abs(h_exact)
        # u_error = np.abs(u_pred_plot - u_exact)

        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Water height
        ax1.plot(x_np, h_pred_plot, 'b-', label='PINN h(x,t)', linewidth=2)
        #ax1.plot(x_np, eta_plot, 'm--', label='Free surface η = h + zb', linewidth=2)
        ax1.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
        #ax1.plot(x_np, Fr_plot, 'm-', label='Froude number', linewidth=1.5)
        ax1.plot(x_np, h_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
        #ax1.axvline(x=dam_position, color='k', linestyle=':', alpha=0.5, label='Dam position')
        ax1.set_xlabel('x [m]')
        ax1.set_ylabel('Water Height h(x) [m]')
        ax1.set_title(f'Water Height')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        #ax1.set_ylim(0, eta_val * 1.5)
        
        # Velocity
        ax2.plot(x_np, u_pred_plot, 'b-', label='PINN u(x,t)', linewidth=2)
        ax2.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
        #ax2.plot(x_np, q_val / h_pred_plot, 'm-', label='Derived u(x,t)', linewidth=2)
        #ax2.plot(x_np, u_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
        ax2.set_xlabel('x [m]')
        ax2.set_ylabel('Velocity u(x) [m/s]')
        ax2.set_title(f'Velocity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        #Froude number plot
        ax3.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
        ax3.plot(x_np, Fr_plot, 'b-', label='Froude number', linewidth=1.5)
        ax3.plot(x_np, fr_critical_mask_plot, 'r-', label='Froude number mask', linewidth=1.5)
        ax3.set_xlabel('x [m]')
        ax3.set_ylabel('Froude number')
        ax3.set_title(f'Froude number')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot height error
        ax4.plot(x_np, h_error, 'r-', label='h_pred - h_exact', linewidth=1.5)
        ax4.plot(x_np, h_error_ratio, 'b-', label = '(h_pred - h_exact)/h_exact', linewidth=1.5)
        #ax4.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
        ax4.set_xlabel('x [m]')
        ax4.set_ylabel('|h_pred - h_exact|')
        ax4.set_title(f'Height Error - Max: {np.max(h_error):.4f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        #ax4.set_yscale('log')

        plt.tight_layout()
        
        # Add information text
        info_text = f"Epochs: {epochs} | Points: {num_collocation_points}\n"
        info_text += f"Training time: {elapsed_time:.1f}s | Final loss: {loss.item():.4e}"
        plt.figtext(0.5, 0.0, info_text, ha='center', fontsize=9)
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

    plt.figure(figsize=(10, 6))
    plt.semilogy(pde_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('pde_loss')
    plt.title('PDE Loss History')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pde_loss_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(continuity_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('continuity_loss')
    plt.title('Continuity Loss History')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'continuity_loss_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(momentum_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('momentum_loss')
    plt.title('Momentum Loss History')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'momentum_loss_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.semilogy(q_loss_history, 'b-', linewidth=2)
    # plt.xlabel('Epoch')
    # plt.ylabel('q_loss')
    # plt.title('Discharge(q) Loss History')
    # plt.grid(True, alpha=0.3)
    # plt.savefig(os.path.join(output_dir, 'q_loss_history.png'), dpi=150, bbox_inches='tight')
    # plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_constraint_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('loss_constraint')
    plt.title('loss_constraint_history')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_constraint_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_constraint_velocity_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('loss_constraint_velocity')
    plt.title('loss_constraint_velocity_history')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_constraint_velocity_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_constraint_height_before_bump_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('loss_constraint_height_before_bump')
    plt.title('loss_constraint_height_before_bump_history')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_constraint_height_before_bump_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_constraint_height_after_hydraulic_jump_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('loss_constraint_height_after_hydraulic_jump')
    plt.title('loss_constraint_height_after_hydraulic_jump_history')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_constraint_height_after_hydraulic_jump_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_constraint_velocity_after_hydraulic_jump_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('loss_constraint_velocity_after_hydraulic_jump')
    plt.title('loss_constraint_velocity_after_hydraulic_jump_history')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_constraint_velocity_after_hydraulic_jump_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSolution images saved in '{output_dir}'")
    torch.save(model.state_dict(), os.path.join(output_dir, 'curriculum_swe_pinn_model.pth'))
    print("Model saved successfully!")

def logdata(x,h,u,q,hu_x,flux_x,bed_momentum):
    with open("x.txt", "w") as file:
        for row in x.unbind(0):
            file.write(str(row.item()))
            file.write("\n")

    with open("h.txt", "w") as file:
        for row in h.unbind(0):
            file.write(str(row.item()))
            file.write("\n")

    with open("u.txt", "w") as file:
        for row in u.unbind(0):
            file.write(str(row.item()))
            file.write("\n")
    
    with open("q.txt", "w") as file:
        for row in q.unbind(0):
            file.write(str(row.item()))
            file.write("\n")
    
    with open("hu_x.txt", "w") as file:
        for row in hu_x.unbind(0):
            file.write(str(row.item()))
            file.write("\n")
    
    with open("flux_x.txt", "w") as file:
        for row in flux_x.unbind(0):
            file.write(str(row.item()))
            file.write("\n")

    with open("bed_momentum.txt", "w") as file:
        for row in bed_momentum.unbind(0):
            file.write(str(row.item()))
            file.write("\n")

def get_analytical_results(case=6):
    os.chdir(dire)
    if case==6:
        eta = np.loadtxt('eta_supercritical.dat')
    if case==7:
        eta = np.loadtxt('eta_subcritical.dat')
    print("loaded analytical results!")

    m=50 #In HLLC simulation 51st record is the last time snapshot state of eta 
    return eta[m, :]

def plot_analytical():
    os.chdir(dire)
    # ========================== Load Data Files subcritical==============================
    print("Loading files...")
    ic = np.loadtxt('Slope_test_subcritical.ic')
    eta = np.loadtxt('eta_subcritical.dat')
    q = np.loadtxt('q_subcritical.dat')
    print("loaded!")
    m, n = q.shape
    x = ic[:, 0]
    zb = ic[:, 1]
    dx = x[-1] - x[-2]
    xf = x[-1] + dx
    eta_up = np.max(eta) + 0.2 * np.max(eta)

    print("plotting...")
    tsim = 0
    m=50 #In HLLC simulation 51st record is the last time snapshot state of eta 
    plt.figure()
    plt.plot(x, zb, 'g--', linewidth=1.5)
    plt.title(f"Free surface evolution")
    plt.plot(x, eta[m, :], 'b-', linewidth=1.5)
    plt.ylim([0, eta_up])
    plt.xlim([x[0], xf])
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.grid()
    plt.savefig(os.path.join(dire, 'hllc_solution_subcritical.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================== Load Data Files supercritical==============================
    print("Loading supercritical files...")
    ic = np.loadtxt('Slope_test_supercritical.ic')
    eta = np.loadtxt('eta_supercritical.dat')
    q = np.loadtxt('q_supercritical.dat')
    print("loaded supercritical!")
    m, n = q.shape
    x = ic[:, 0]
    zb = ic[:, 1]
    dx = x[-1] - x[-2]
    xf = x[-1] + dx
    eta_up = np.max(eta) + 0.2 * np.max(eta)

    print("plotting...")
    tsim = 0
    m=50 #In HLLC simulation 51st record is the last time snapshot state of eta 
    plt.figure()
    plt.plot(x, zb, 'g--', linewidth=1.5)
    plt.title(f"Free surface evolution")
    plt.plot(x, eta[m, :], 'b-', linewidth=1.5)
    plt.ylim([0, eta_up])
    plt.xlim([x[0], xf])
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.grid()
    plt.savefig(os.path.join(dire, 'hllc_solution_supercritical.png'), dpi=150, bbox_inches='tight')
    plt.close()

def getxplot(case):
    os.chdir(dire)
    if case==6:
        ic = np.loadtxt('Slope_test_supercritical.ic')
    if case==7:
        ic = np.loadtxt('Slope_test_subcritical.ic')
    x = ic[:, 0]
    x1 = np.float32(x)
    return torch.from_numpy(x1).view(-1, 1)

def main():
    start_time = time.time()
    global test_case #supercritical:6 subcritical:7
    test_case = 7
    set_eta_q(test_case)
    loss = train()
    test(loss,start_time)
    #plot_analytical()

if __name__=="__main__":
    main()
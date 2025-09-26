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
epochs = 2000
learning_rate = 1e-3
num_time_steps = 20
#Scheduler tuning parameters
scheduler_step_size_frequency = 1 #Number of times we want scheduler to reduce LR during full training with epochs
scheduler_step_size = epochs // scheduler_step_size_frequency # Epoch intervals at which scheduler will reduce LR 
scheduler_gamma=0.9 #Factor by which scheduler will reduce LR at each epoch interval
# Initial condition parameters
eta_val = 2.0
q_val = 4.42

# Output directory
output_dir = "swe/temp/dualhead/swe_case7_" + str(epochs)
os.makedirs(output_dir, exist_ok=True)

loss_history = []
pde_loss_history = []
momentum_loss_history = []
continuity_loss_history = []
q_loss_history = []
gradient = []

def get_weights(epoch, total_epochs):
    # start with strong BC enforcement, gradually relax
    bc_weight = 500.0 if epoch < 1000 else 50.0
    pde_weight = 100.0
    return pde_weight, bc_weight

def get_weights_pde(epoch, total_epochs,near_bump=False):
    if near_bump==False:
        continuity_loss_weight = 5.0 if epoch < 2000 else 5.0
        momentum_loss_weight = 1.0 if epoch < 2000 else 1.0
    else:
        continuity_loss_weight = 10.0
        momentum_loss_weight = 1.0
    return continuity_loss_weight, momentum_loss_weight

class ImprovedPINN_SWE(nn.Module):
    """
    Improved Physics-Informed Neural Network for 1D Shallow Water Equations
    """
    def __init__(self):
        super(ImprovedPINN_SWE, self).__init__()
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
        features = self.backbone(x)
        h_raw = self.h_head(features)
        u_raw = self.u_head(features)
        epsilon = 1e-3
        return torch.clamp(h_raw, min=epsilon), u_raw

# Instantiate the network
model = ImprovedPINN_SWE()

# Optimizer with scheduled learning rate
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, scheduler_step_size, scheduler_gamma)

def improved_physics_loss(h, u, x, epoch,epochs,near_bump=False):
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
    dzb_dx = torch.autograd.grad(zb, x, grad_outputs=torch.ones_like(zb), create_graph=True)[0]

    #friction slope
    # manning = 0.04
    # sfx = manning**2 * u * torch.abs(u) / h.clamp(min=1e-3)**(4.0/3.0)
    momentum_residual = flux_x + g * h * dzb_dx #+ g * h * sfx

    # explicit q constraint — helps enforce constant discharge
    bump_mask = ((x > 8.0) & (x < 12.0)).float()
    outside_bump_mask = 1.0 - bump_mask
    q_loss_bump_region = torch.mean((q * bump_mask.float() - q_val * bump_mask.float())**2)
    #q_loss = torch.mean((q * outside_bump_mask.float() - q_val)**2)

    # Weighted PDE residuals
    continuity_loss = torch.mean(continuity_residual**2)
    momentum_loss = torch.mean(momentum_residual**2)

    continuity_loss_weight_curr, momentum_loss_weight_curr = get_weights_pde(epoch, epochs,near_bump)

    # Combine total PDE loss
    total_pde_loss = continuity_loss_weight_curr * continuity_loss \
        + momentum_loss_weight_curr * momentum_loss \
        + q_loss_bump_region \
        #+ q_loss

    return total_pde_loss, {
        'continuity': continuity_loss.item(),
        'momentum': momentum_loss.item(),
        'q_loss': q_loss_bump_region.item()
    }

# ------------------ Bed Elevation Function ------------------

def bed_elevation(x: torch.Tensor) -> torch.Tensor:
    zb = torch.zeros_like(x)
    zb_h = 0.2 - 0.05 * (x - 10.0) **2
    return torch.where((x > 8.0) & (x < 12.0), zb_h, zb)

def set_eta_q(case=7):
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

    loss_bc_left = torch.mean((h_left_pred - h_bc_left())**2) + 10 * torch.mean((u_left_pred - u_bc_left())**2)
    loss_bc_right = torch.mean((h_right_pred - h_bc_right())**2) + 10 * torch.mean((u_right_pred - u_bc_right())**2)
    return  (loss_bc_left + loss_bc_right)

def train():
    # Generate training data
    # Collocation points
    #x_collocation = torch.rand(int(num_collocation_points), 1) * (x_max - x_min) + x_min

    # Enhanced training data sampling
    # Much denser sampling near bump
    x_bump_region = torch.linspace(8, 12, num_collocation_points // 4)
    x_outer_left = torch.linspace(x_min, 8, num_collocation_points // 3)
    x_outer_right = torch.linspace(12, x_max, num_collocation_points // 3)
    x_collocation = torch.cat([x_outer_left, x_bump_region, x_outer_right]).reshape(-1, 1)
    
    # Boundary points
    x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
    x_boundary_right = torch.ones(num_boundary_points, 1) * x_max

    # Set requires_grad
    x_collocation.requires_grad_(True)
    x_boundary_left.requires_grad_(True)
    x_boundary_right.requires_grad_(True)

    print(f"Domain: x ∈ [{x_min}, {x_max}]")

    start_time = time.time()
    model.train()

    #Setting test case here. 
    set_eta_q(7)

    # Training loop
    for epoch in range(epochs):
        loss, loss_pde, loss_boundary, pde_components = \
            compute_total_loss(x_collocation,x_boundary_left,x_boundary_right,epoch,epochs)
        # # Physics loss
        # h_collocation, u_collocation = model(x_collocation)
        # loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
        #                                                 x_collocation,epoch,epochs)
    
        # # Boundary loss
        # h_boundary_left, u_boundary_left = model(x_boundary_left)
        # h_boundary_right, u_boundary_right = model(x_boundary_right)
        # loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
        #                                     h_boundary_right, u_boundary_right)

        # lambda_pde_curr, lambda_bc_curr = get_weights(epoch, epochs)

        # # explicit q constraint — helps enforce constant discharge
        # # q = h_collocation * u_collocation
        # # q_loss = torch.mean((q - q_val)**2)

        # # Total loss
        # loss = (
        #     lambda_pde_curr * loss_pde
        #     + lambda_bc_curr * loss_boundary
        # )


        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        pde_loss_history.append(loss_pde.item())
        continuity_loss_history.append(pde_components['continuity'])
        momentum_loss_history.append(pde_components['momentum'])
        q_loss_history.append(pde_components['q_loss'])

        # optional: add small parameter noise to escape local minima
        if (epoch + 1) % 200 == 0 and epoch > 0:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(1e-5 * torch.randn_like(p))

        # after optimizer step or before next iter
        # for p in model.parameters():
        #     p.grad += 1e-6 * torch.randn_like(p.grad)  # smaller magnitude for stability

        # # optional: add small param noise every N epochs
        # if epoch % 200 == 0:
        #     for p in model.parameters():
        #         p.data += 1e-5 * torch.randn_like(p.data)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {loss.item():.4e}")
            print(f"  PDE Loss: {loss_pde.item():.4e}")
            print(f"  Boundary Loss: {loss_boundary.item():.4e}")
            print(f"  Continuity: {pde_components['continuity']:.4e}")
            print(f"  Momentum: {pde_components['momentum']:.4e}")
            print(f"  Discharge(q_loss): {pde_components['q_loss']:.4e}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            calculate_gradient_norm()


    # ---- LBFGS refinement phase ----
    opt_lbfgs = optim.LBFGS(model.parameters(),
                            max_iter=500,
                            history_size=50,
                            tolerance_grad=1e-9,
                            tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")
    
    def closure():
        opt_lbfgs.zero_grad()
        loss, loss_pde, loss_boundary, pde_components = \
            compute_total_loss(x_collocation,x_boundary_left,x_boundary_right,epoch,epochs)
        loss.backward()
        loss_history.append(loss.item())
        pde_loss_history.append(loss_pde.item())
        continuity_loss_history.append(pde_components['continuity'])
        momentum_loss_history.append(pde_components['momentum'])
        q_loss_history.append(pde_components['q_loss'])
        return loss

    print("Switching to LBFGS...")
    opt_lbfgs.step(closure)

    # Final loss check
    loss, loss_pde, loss_boundary, pde_components = \
        compute_total_loss(x_collocation,x_boundary_left,x_boundary_right,epoch,epochs)
    
    print(f"  Total Loss: {loss.item():.4e}")
    print(f"  PDE Loss: {loss_pde.item():.4e}")
    print(f"  Boundary Loss: {loss_boundary.item():.4e}")
    print(f"  Continuity: {pde_components['continuity']:.4e}")
    print(f"  Momentum: {pde_components['momentum']:.4e}")
    print(f"  Discharge(q_loss): {pde_components['q_loss']:.4e}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    return loss

def compute_total_loss(x_collocation,x_boundary_left,x_boundary_right,epoch,epochs):
    # Physics loss
    h_collocation, u_collocation = model(x_collocation)
    loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                    x_collocation,epoch,epochs)

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
    return loss, loss_pde, loss_boundary, pde_components


def calculate_gradient_norm():
    # Calculate the average gradient norm
    total_norm = 0
    num_parameters = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2) # Calculate L2 norm
            total_norm += param_norm.item()
            num_parameters += 1

    if num_parameters > 0:
        average_gradient_norm = total_norm / num_parameters
        print(f"  Average Gradient Norm: {average_gradient_norm}")
        gradient.append(average_gradient_norm)
    else:
        print("No parameters with gradients found.")


def train_near_bump():
    # Generate training data
    # Collocation points
    x_collocation = torch.rand(int(num_collocation_points), 1) * (12 - 8) + 8

    # Boundary points
    x_boundary_left = torch.ones(num_boundary_points, 1) * x_min
    x_boundary_right = torch.ones(num_boundary_points, 1) * x_max

    # Set requires_grad
    x_collocation.requires_grad_(True)
    x_boundary_left.requires_grad_(True)
    x_boundary_right.requires_grad_(True)

    print(f"Near Bump Training - Domain: x ∈ [{x_min}, {x_max}]")

    start_time = time.time()
    model.train()

    #Setting test case here. 
    set_eta_q(7)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Physics loss
        h_collocation, u_collocation = model(x_collocation)
        loss_pde, pde_components = improved_physics_loss(h_collocation, u_collocation, 
                                                        x_collocation,epoch,epochs,True)
    
        # Boundary loss
        h_boundary_left, u_boundary_left = model(x_boundary_left)
        h_boundary_right, u_boundary_right = model(x_boundary_right)
        loss_boundary = boundary_condition_loss(h_boundary_left, u_boundary_left, 
                                            h_boundary_right, u_boundary_right)

        #lambda_pde_curr, lambda_bc_curr = get_weights(epoch, epochs)

        # Total loss
        loss = (
            loss_pde
            + loss_boundary
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
        q_loss_history.append(pde_components['q_loss'])
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss: {loss.item():.4e}")
            print(f"  PDE Loss: {loss_pde.item():.4e}")
            print(f"  Boundary Loss: {loss_boundary.item():.4e}")
            print(f"  Continuity: {pde_components['continuity']:.4e}")
            print(f"  Momentum: {pde_components['momentum']:.4e}")
            print(f"  Discharge(q_loss): {pde_components['q_loss']:.4e}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    elapsed_time = time.time() - start_time
    print(f"Near Bump Training completed in {elapsed_time:.2f} seconds.")
    return loss

def test(loss,start_time):
    elapsed_time = time.time() - start_time
    # Generate solution plots
    x_plot = torch.linspace(x_min, x_max, 500).view(-1, 1)

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
        #ax1.set_ylim(0, eta_val * 1.5)
        
        # Velocity
        ax2.plot(x_np, u_pred_plot, 'b-', label='PINN u(x,t)', linewidth=2)
        #ax2.plot(x_np, q_val/h_pred_plot, 'm-', label='Derived u(x,t)', linewidth=2)
        ax2.plot(x_np, zb_plot, 'g--', label='Bottom topography zb(x)', linewidth=1.5)
        #ax2.plot(x_np, u_exact, 'r--', label='Analytical solution', linewidth=2, alpha=0.8)
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

    plt.figure(figsize=(10, 6))
    plt.semilogy(q_loss_history, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('q_loss')
    plt.title('Discharge(q) Loss History')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'q_loss_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(gradient, 'b-', linewidth=2)
    plt.xlabel('Epoch( x 100)')
    plt.ylabel('gradient')
    plt.title('Gradient History')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gradient_history.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSolution images saved in '{output_dir}'")
    torch.save(model.state_dict(), os.path.join(output_dir, 'curriculum_swe_pinn_model.pth'))
    print("Model saved successfully!")

def main():
    start_time = time.time()
    loss = train()
    #loss = train_near_bump()
    #loss = train()
    test(loss,start_time)

if __name__=="__main__":
    main()
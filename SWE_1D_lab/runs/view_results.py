import os
import numpy as np
import matplotlib.pyplot as plt

# ========================== Configuration ===============================
dire = 'C:\\Users\\rhear\\MPAS\\PINN\\SWE_1D_lab\\runs' # Update this path
plot_anim = True
plot_v = True
plot_q = True
plot_sensors = False
tpausa = 0.1

# ========================== Load Simulation Settings =====================
os.chdir(dire)

with open('settings_sim.inp', 'r') as f:
    lines = f.readlines()
initial_cond_file = lines[6].strip()
dt = float(lines[10].strip())
nsens = int(lines[14].strip())

# ========================== Load Data Files ==============================
print("Loading files...")
eta = np.loadtxt('eta.dat')
ic = np.loadtxt(initial_cond_file)
q = np.loadtxt('q.dat')
h = np.loadtxt('h.dat')
print("loaded!")

m, n = q.shape
x = ic[:, 0]
zb = ic[:, 1]
dx = x[-1] - x[-2]
xf = x[-1] + dx
eta_up = np.max(eta) + 0.2 * np.max(eta)

# ========================== Plot Free Surface ============================
if plot_anim:
    print("plotting...")
    tsim = 0
    plt.figure()
    for i in range(m):
        plt.plot(x, zb, '-k', linewidth=2)
        tsim += dt
        plt.title(f"Free surface evolution, t = {tsim:.2f} s")
        plt.plot(x, eta[i, :], '+b', linewidth=2)
        plt.ylim([0, eta_up])
        plt.xlim([x[0], xf])
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.grid()
        plt.pause(tpausa)
        plt.clf()

# ========================== Plot Velocity ================================
if plot_v:
    print("plotting velocities...")
    v = np.where(h > 0, q / h, 0)
    v_up = np.max(v) + 0.2 * np.max(v)
    v_min = np.min(v) - 0.2 * np.min(v)

    plt.figure()
    tsim = 0
    for j in range(m):
        plt.plot(x, v[j, :], '-r', linewidth=2)
        tsim += dt
        plt.title(f"Velocity, t = {tsim:.2f} s")
        plt.ylim([v_min, v_up])
        plt.xlim([x[0], xf])
        plt.xlabel('x (m)')
        plt.ylabel('velocity (m/s)')
        plt.grid()
        plt.pause(tpausa)
        plt.clf()

    plt.plot(x, v[0, :], '--k', linewidth=2)
    plt.plot(x, v[-1, :], '-b', linewidth=2)

# ========================== Plot Discharge ===============================
if plot_q:
    print("plotting discharge...")
    q_up = np.max(q) + 0.2 * np.max(q)
    q_min = np.min(q) - 0.2 * np.min(q)

    plt.figure()
    tsim = 0
    for j in range(m):
        plt.plot(x, q[j, :], '*m', linewidth=2)
        tsim += dt
        plt.title(f"Discharge, t = {tsim:.2f} s")
        plt.ylim([q_min, q_up])
        plt.xlim([x[0], xf])
        plt.xlabel('x (m)')
        plt.ylabel('discharge (m^2/s)')
        plt.grid()
        plt.pause(tpausa)
        plt.clf()

# ========================== Plot Sensors ================================
if plot_sensors:
    if nsens == 0:
        raise RuntimeError('No sensors recording according to input file')
    else:
        print("plotting sensors time series...")
        s = np.loadtxt('sensors.dat')
        labels = []

        plt.figure()
        plt.subplot(1, 2, 1)
        for i in range(1, nsens + 1):
            plt.plot(s[:, 0], s[:, i])
        plt.title('Water depth')
        plt.xlabel('time (s)')
        plt.ylabel('water depth (m)')

        plt.subplot(1, 2, 2)
        for j in range(nsens):
            plt.plot(s[:, 0], s[:, nsens + 1 + j])
            labels.append(f"sensor {j + 1}")
        plt.title('Discharge')
        plt.xlabel('time (s)')
        plt.ylabel('discharge (m^2/s)')
        plt.legend(labels)

        plt.tight_layout()
        plt.show()

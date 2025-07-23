from fix_pathing import root_dir

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
from multiprocessing import Manager, Pool, cpu_count
import time
import pickle

from src.initial_state import checkerboard_state, empty_state, random_state, product_state
from src.simulation import trajectory
from src.setup import construct_H, get_bonds

plotting = True
saving = not plotting

plotting_threshold = 0.00  # Threshold for plotting currents


num_processes = cpu_count() - 1  # Leave one core free for the OS
if num_processes < 1:
    raise ValueError("Not enough CPU cores available for multiprocessing.")

dt = 0.1
p = 0.15

drive_type = "current"  # "current", "dephasing"
corner_dephasing = False  # Whether to apply dephasing at the corners

Nx = 3
Ny = 3
N = Nx*Ny

b = 0.0 #2/((Nx-1)*(Ny-1))  # Magnetic field strength
B = b*np.pi # Magnetic field in units of flux quantum

t = 0.0  # Diagonal hopping

num_iterations = 50000

# Spread iterations across the available processes
batch_size = [num_iterations // num_processes]*num_processes
# Spread any remaining iterations across the first few processes
for i in range(num_iterations % num_processes):
    batch_size[i] += 1



steps = 500
initial_state = "random"  # "checkerboard", "empty", "random", "custom"

even_parity = False  # Only used for random state
occupation_list = [0,1,0,1,1,0,1,0,0]  # Only used for custom state

t_list = np.linspace(0, steps*dt, steps+1)
# t_list = np.arange(0, (steps+1)*dt, dt)


site_in = 0  # Site where the current is injected
site_out = N-1  # Site where the current is extracted


if not drive_type in ["current", "dephasing"]:
    raise ValueError(f"Invalid drive_type: {drive_type}")

if not initial_state in ["checkerboard", "empty", "random", "custom"]:
    raise ValueError(f"Invalid initial_state: {initial_state}")


if __name__ == "__main__":

    H = construct_H(Nx, Ny)
    U = expm(-1j*H*dt)

    bonds = get_bonds(Nx, Ny, site_in, site_out, t=t)

    alpha = None
    if initial_state == "checkerboard":
        alpha = checkerboard_state(Nx, Ny)
    elif initial_state == "empty":
        alpha = empty_state(Nx, Ny)
    elif initial_state == "random":
        alpha = random_state(Nx, Ny, even_parity=even_parity)
    elif initial_state == "custom":
        alpha = product_state(occupation_list, Nx, Ny)


    manager = Manager()
    data = manager.dict()
    data["H"] = H
    data["U"] = U
    data["alpha"] = alpha
    data["completed"] = 0



    print(f"CPU count: {cpu_count()}")

    K_avg = 0.
    n_avg = 0.
    avg_currents = 0.
    avg_dd_correlations = 0.

    t1 = time.perf_counter()


    with Pool(processes=num_processes) as pool:
        for procid in range(num_processes):
            pool.apply_async(trajectory, args=(procid, data, steps, batch_size[procid], Nx, Ny, p, bonds, site_in, site_out, drive_type, corner_dephasing, initial_state))
        pool.close()
        pool.join()

    # for procid in range(num_iterations):
    #     trajectory(procid, data, steps, batch_size[procid], Nx, Ny, p, bonds, site_in, site_out, drive_type, corner_dephasing, initial_state)
        

    for i in range(num_processes):
        res = data[i]
        K_avg += res["K_list"] / num_iterations
        n_avg += res["n_list"] / num_iterations
        avg_currents += res["currents_list"] / num_iterations
        avg_dd_correlations += res["density_correlations"] / num_iterations

    t2 = time.perf_counter()
    print(f"\n Finished all trajectories")
    print(f"Time taken (parallel): {t2 - t1} seconds")


    for i in range(N):
        plt.plot(t_list, n_avg[:,i], label=f"n_{i}")
    plt.plot(t_list, np.sum(n_avg, axis=1)/N, '--', c='k', label="n_avg")
    plt.xlabel("Time")
    plt.ylabel("Occupation")
    # plt.legend()
    # plt.xscale("log")
    plt.show()
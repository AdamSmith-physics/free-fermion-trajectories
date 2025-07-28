import numpy as np
from .initial_state import random_state
from .parameter_dataclasses import SimulationParameters
from tqdm import tqdm

def apply_n(alpha, i, N):

    alpha_cut = np.abs(alpha[i+N,:])

    i0 = np.argmax(alpha_cut)
    # move i_0 column of alpha to end
    alpha = np.roll(alpha, -i0-1, axis=1)

    alpha_i0 = alpha[:,-1]

    alpha_new = np.zeros((2*N,N), dtype=complex)

    # vectorized
    alpha_new[:,:-1] = alpha[:,:-1] - (alpha[i+N,:-1]/alpha_i0[i+N])*alpha_i0.reshape((2*N,1))
    alpha_new[i,:-1] = 0

    alpha_new[:,-1] = np.zeros((2*N,))
    alpha_new[i+N, -1] = 1

    alpha_new, _ = np.linalg.qr(alpha_new)
    
    alpha_new[:,-1] = np.zeros((2*N,))
    alpha_new[i+N, -1] = 1

    return alpha_new


def apply_1_minus_n(alpha, i, N):
    alpha_cut = np.abs(alpha[i,:])

    i0 = np.argmax(alpha_cut)
    # move i_0 colum of alpha to end
    alpha = np.roll(alpha, -i0-1, axis=1)

    alpha_i0 = alpha[:,-1]

    alpha_new = np.zeros((2*N,N), dtype=complex)

    # vectorized
    alpha_new[:,:-1] = alpha[:,:-1] - (alpha[i,:-1]/alpha_i0[i])*alpha_i0.reshape((2*N,1))
    alpha_new[i+N,:-1] = 0

    alpha_new[:,-1] = np.zeros((2*N,))
    alpha_new[i, -1] = 1

    alpha_new, _ = np.linalg.qr(alpha_new)
    
    alpha_new[:,-1] = np.zeros((2*N,))
    alpha_new[i, -1] = 1

    return alpha_new


def apply_cn(alpha, i, N):
    alpha = apply_n(alpha, i, N)
    alpha[:,-1] = np.zeros((2*N,))
    alpha[i, -1] = 1

    return alpha


def apply_cdag_1_minus_n(alpha, i, N):
    alpha = apply_1_minus_n(alpha, i, N)
    alpha[:,-1] = np.zeros((2*N,))
    alpha[i+N, -1] = 1

    return alpha


def pick_kraus(C, p, N):

    p_capped = max(0, min(p, 1))  # Ensure p is between 0 and 1

    n_in = np.real(C[N,N])
    n_out = np.real(C[2*N-1, 2*N-1])

    probs_in = np.cumsum([1-p_capped, p_capped*(1-n_in), p_capped*n_in])
    probs_out = np.cumsum([1-p_capped, p_capped*(1-n_out), p_capped*n_out])

    coins = np.random.rand(2)

    K_in = sum(coins[0] > probs_in)
    K_out = sum(coins[1] > probs_out)

    return K_in, K_out


def trajectory(procid, data, batch_size, steps, params: SimulationParameters):

    # Unpack parameters ######
    steps = params.steps
    Nx = params.Nx
    Ny = params.Ny
    p = params.p
    bonds = params.bonds
    site_in = params.site_in
    site_out = params.site_out
    drive_type = params.drive_type
    corner_dephasing = params.corner_dephasing
    initial_state = params.initial_state
    ###########################

    N = Nx*Ny

    K_list_accumulated = np.zeros((steps, 9), dtype=int)
    n_list_accumulated = np.zeros((steps+1, N), dtype=float)
    currents_list_accumulated = np.zeros((steps+1, len(bonds)), dtype=float)
    density_correlations_accumulated = np.zeros((N, N), dtype=float)


    for run in range(batch_size):

        H = data["H"].copy()
        U = data["U"].copy()
        if initial_state == "random":
            alpha = random_state(Nx, Ny)  # Use a new random state for each trajectory
        else:
            alpha = data["alpha"].copy()

        C = alpha @ alpha.T.conj() 

        K_list = np.zeros((steps,9), dtype=int)
        n_list = np.zeros((steps+1, N), dtype=float)
        currents_list = np.zeros((steps+1, len(bonds)), dtype=float)
        
        n_list[0] = [np.real(C[N+i,N+i]) for i in range(N)]
        currents_list[0] = [np.real(1j*(H[n1,n2]*C[n1+N,n2+N] - H[n2,n1]*C[n2+N,n1+N])) for n1, n2 in bonds]
        for step in range(steps):

            alpha = U @ alpha

            C = alpha @ alpha.T.conj()
            K_in, _ = pick_kraus(C, p, N)

            if K_in == 1:
                if drive_type == "current":
                    alpha = apply_cdag_1_minus_n(alpha, site_in, N)
                elif drive_type == "dephasing":
                    alpha = apply_1_minus_n(alpha, site_in, N)
            elif K_in == 2:
                alpha = apply_n(alpha, site_in, N)

            
            C = alpha @ alpha.T.conj()
            _, K_out = pick_kraus(C, p, N)

            if K_out == 1:
                alpha = apply_1_minus_n(alpha, site_out, N)
            elif K_out == 2:
                if drive_type == "current":
                    alpha = apply_cn(alpha, site_out, N)
                elif drive_type == "dephasing":
                    alpha = apply_n(alpha, site_out, N)

            K_list[step, K_in + 3*K_out] = 1
            

            if corner_dephasing and step > steps//2:
                C = alpha @ alpha.T.conj()
                if np.random.rand(1) < p: 
                    n_corner = C[N+1, N+1]
                    if np.random.rand(1) < n_corner:
                        alpha = apply_n(alpha, 1, N)
                    else:
                        alpha = apply_1_minus_n(alpha, 1, N)


            C = alpha @ alpha.T.conj()

            n_list[step+1] = [np.real(C[N+i,N+i]) for i in range(N)]

            currents_list[step+1] = [np.real(1j*(H[n1,n2]*C[n1+N,n2+N] - H[n2,n1]*C[n2+N,n1+N])) for n1, n2 in bonds]

            if step == steps - 1:
                # Compute the density-density correlation matrix using Wick's theorem
                density_correlations = np.zeros((N, N))
                for n1 in range(N):
                    for n2 in range(N):
                        density_correlations[n1, n2] = -np.abs(C[N+n1, N+n2])**2 - np.abs(C[N+n1, n2])**2  # Connected density-density correlations

        K_list_accumulated += K_list
        n_list_accumulated += n_list
        currents_list_accumulated += currents_list
        density_correlations_accumulated += density_correlations

        if procid == 0:
            print(f"Process {procid}, run {run+1}/{batch_size} completed", end="\r")

    trajectory_data = {"K_list": K_list_accumulated, "n_list": n_list_accumulated, "currents_list": currents_list_accumulated, "density_correlations": density_correlations_accumulated}

    # data["completed"] += 1
    # print(f"Completed {data['completed']}", end="\r") # May not give correct value!

    data[procid] = trajectory_data

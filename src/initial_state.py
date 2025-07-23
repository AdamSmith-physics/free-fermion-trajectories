import numpy as np

def product_state(occupation_list, Nx, Ny):
    """
    Create a product state given a list of occupations.
    Each occupation is either 0 (empty) or 1 (filled).
    """
    alpha_list = []
    N = Nx * Ny

    if len(occupation_list) != N:
        raise ValueError(f"Occupation list must have length {N}, got {len(occupation_list)}")
    if not all(occ in [0, 1] for occ in occupation_list):
        raise ValueError("Occupation list must contain only 0s and 1s")


    alpha = np.zeros((2*N, N), dtype=complex)
    for n, occupation in enumerate(occupation_list):
        alpha[:,n] = [0]*(2*Nx*Ny)
        if occupation == 1:
            alpha[n + Nx*Ny,n] = 1.0
        else:
            alpha[n,n] = 1.0

    return alpha


def checkerboard_state(Nx, Ny):
    occupation_list = []
    # Create a checkerboard pattern of occupations
    for y in range(Ny):
        for x in range(Nx):
            if (x + y) % 2 == 1:
                occupation_list.append(1)  # filled site
            else:
                occupation_list.append(0)  # empty site

    return product_state(occupation_list, Nx, Ny)


def empty_state(Nx, Ny):
    occupation_list = [0] * (Nx * Ny)  # all sites empty
    return product_state(occupation_list, Nx, Ny)


def random_state(Nx, Ny, even_parity=False):
    occupation_list = np.random.choice([0, 1], size=(Nx * Ny,), p=[0.5, 0.5])
    if even_parity:
        occupation_list[-1] = 1 - occupation_list[-1] # Ensure even parity by flipping the last site if necessary
    return product_state(occupation_list, Nx, Ny)
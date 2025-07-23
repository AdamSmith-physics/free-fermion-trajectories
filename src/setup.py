import numpy as np

def get_bonds(Nx, Ny, site_in, site_out, t=0.0):
    """
    Generate a list of bonds for a 2D lattice with specified dimensions and boundary conditions.
    
    Parameters:
    Nx (int): Number of sites in the x-direction.
    Ny (int): Number of sites in the y-direction.
    site_in (int): Index of the input site.
    site_out (int): Index of the output site.
    t (float): Hopping parameter for diagonal bonds. Default is 0.0.
    
    Returns:
    list: A list of tuples representing the bonds between sites.
    """

    # Extract currents
    bonds = []
    for y in range(Ny):
        for x in range(Nx-1):
            n1 = x % Nx + y*Nx
            n2 = (x+1) % Nx + y*Nx

            if not n1 in [site_in, site_out] and not n2 in [site_in, site_out]:
                bonds.append((n1, n2))

    for x in range(Nx):
        for y in range(Ny-1):
            n1 = x % Nx + y*Nx
            n2 = x % Nx + (y+1)*Nx

            if not n1 in [site_in, site_out] and not n2 in [site_in, site_out]:
                bonds.append((n1, n2))

    if t != 0.0:
        for y in range(1, Ny):
            for x in range(Nx-1):
                n1 = x % Nx + y*Nx
                n2 = (x+1) % Nx + (y-1)*Nx

                if not n1 in [site_in, site_out] and not n2 in [site_in, site_out]:
                    bonds.append((n1, n2))

    return bonds


def construct_H(Nx, Ny, B=0.0, t=0.0):

    N = Nx*Ny

    H = np.zeros((N, N), dtype=complex)

    for y in range(Ny):
        for x in range(Nx-1):
            n1 = x % Nx + y*Nx
            n2 = (x+1) % Nx + y*Nx
            H[n1,n2] = -np.exp(1j*B*y)
            H[n2,n1] = -np.exp(-1j*B*y)

    for x in range(Nx):
        for y in range(Ny-1):
            n1 = x % Nx + y*Nx
            n2 = x % Nx + (y+1)*Nx

            H[n1,n2] = -1
            H[n2,n1] = -1

    if t != 0.0:
        for y in range(1, Ny):
            for x in range(Nx-1):
                n1 = x % Nx + y*Nx
                n2 = (x+1) % Nx + (y-1)*Nx

                H[n1,n2] = -t*np.exp(1j*B*(y-1/2))
                H[n2,n1] = -t*np.exp(-1j*B*(y-1/2))

    bigH = np.block([[-H.T, np.zeros((N,N))],
                     [np.zeros((N,N)), H]])

    print("Hamiltonian constructed.")

    return bigH
#quantum walk in 1d
def qw1d(n, coin, initState):
    
    import numpy as np

    L = 2*n + 1
    center = L//2

    psi = np.zeros((L, 2, 1), dtype=complex)
    psi[center] = initState

    x = np.arange(-n, n+1, 1)
    msd_list = []
    prob_list = []

    for _ in range(n):

        for i in range(L):
            psi[i] = np.dot(coin, psi[i])

        psi_prev = psi.copy()
        psi[:] = 0
        psi[1:, 0, 0] = psi_prev [:-1, 0, 0]
        psi[:-1, 1, 0] = psi_prev [1:, 1, 0]

        prob = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
        prob_list.append(prob)
        
        mean = np.sum(x * prob)
        msd = np.sum(x**2 * prob) - mean**2
        #msd = np.sum(x**2 * prob)
        msd_list.append(msd)

    filename = f"qwalk1d_n{n}.npz"

    np.savez(
        filename,
        x=x,
        prob=np.array(prob_list),
        msd=np.array(msd_list)
    )


#quantum walk in 2d
def qw2d(n, coin, initState):
    
    import numpy as np

    L = 2*n + 1
    center = L//2, L//2

    psi = np.zeros((L, L, 4), dtype=complex)
    psi[center] = initState
    
    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    msd_list = []
    prob_list = []

    for _ in range(n):

        for i in range(L):
            for j in range(L):
                psi[i, j] = np.dot(coin, psi[i, j])

        psi_prev = psi.copy()
        psi[:] = 0
        psi[1:, 1:, 0] = psi_prev [:-1, :-1, 0] #up up
        psi[1:, :-1, 1] = psi_prev[:-1, 1:, 1] #up down
        psi[:-1, 1:, 2] = psi_prev[1:, :-1, 2] #down up
        psi[:-1, :-1, 3] = psi_prev[1:, 1:, 3] #down down

        prob = np.sum(np.abs(psi)**2, axis = 2)
        prob_list.append(prob)
        
        Px = prob.sum(axis=0)
        Py = prob.sum(axis=1)

        mean_x = np.sum(x * Px)
        mean_y = np.sum(y * Py)
        msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2)
        #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py)
    
        msd_list.append(msd)

    filename = f"qwalk2d_n{n}.npz"
    np.savez(
        filename,
        x=x,
        y=y,
        prob=np.array(prob_list),
        msd=np.array(msd_list)
    )

#quantum walk in 3d
def qw3d(n, coin, initState):

    import numpy as np

    L = 2*n + 1
    center = L//2, L//2, L//2

    psi = np.zeros((L, L, L, 8), dtype=complex)
    psi[center] = initState

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    z = np.arange(-n, n+1)
    msd_list = []
    prob_list = []
    
    for _ in range(n):

        for i in range(L):
            for j in range(L):
                for k in range(L):
                    psi[i, j, k] = np.dot(coin, psi[i, j, k])

        psi_prev = psi.copy()
        psi[:] = 0
        psi[1:, 1:, 1:, 0] = psi_prev [:-1, :-1, :-1, 0] #up up up
        psi[1:, :-1, 1:, 1] = psi_prev[:-1, 1:, :-1, 1] #up down up
        psi[:-1, 1:, 1:, 2] = psi_prev[1:, :-1, :-1, 2] #down up up
        psi[:-1, :-1, 1:, 3] = psi_prev[1:, 1:, :-1, 3] #down down up
        psi[1:, 1:, :-1, 4] = psi_prev [:-1, :-1, 1:, 4] #up up down
        psi[1:, :-1, :-1, 5] = psi_prev[:-1, 1:, 1:, 5] #up down down
        psi[:-1, 1:, :-1, 6] = psi_prev[1:, :-1, 1:, 6] #down up down
        psi[:-1, :-1, :-1, 7] = psi_prev[1:, 1:, 1:, 7] #down down down

        prob = np.sum(np.abs(psi)**2, axis = 3)
        prob_list.append(prob)

        Px = prob.sum(axis=(1,2))
        Py = prob.sum(axis=(0,2))
        Pz = prob.sum(axis=(0,1))
        mean_x = np.sum(x * Px)
        mean_y = np.sum(y * Py)
        mean_z = np.sum(z * Pz)
        msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2) + (np.sum(z**2 * Pz) - mean_z**2)
        #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py) + np.sum(z**2 * Pz)
    
        msd_list.append(msd)

    filename = f"qwalk3d_n{n}.npz"
    np.savez(
        filename,
        x=x,
        y=y,
        z=z,
        prob=np.array(prob_list),
        msd=np.array(msd_list)
    )

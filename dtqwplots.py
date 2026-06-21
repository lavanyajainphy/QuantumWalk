#quantum walk in 1d
def qw1d(n, coin, initState):
    
    import numpy as np
    import matplotlib.pyplot as plt

    L = 2*n + 1
    center = L//2

    psi = np.zeros((L, 2, 1), dtype=complex)
    psi[center] = initState

    x = np.arange(-n, n+1, 1)
    msd_list = []
    var_list = []

    for _ in range(n):

        for i in range(L):
            psi[i] = np.dot(coin, psi[i])

        psi_prev = psi.copy()
        psi[:] = 0
        psi[1:, 0, 0] = psi_prev [:-1, 0, 0]
        psi[:-1, 1, 0] = psi_prev [1:, 1, 0]

        prob = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
        mean = np.sum(x * prob)
        var = np.sum(x**2 * prob) - mean**2
        var_list.append(var)
        msd = np.sum(x**2 * prob)
        msd_list.append(msd)

    plt.plot(x, prob)
    plt.xlabel("position")
    plt.ylabel("probability")
    plt.title("Probability of Different Positions in 1D Quantum Walk")
    plt.show()

    plt.figure()
    plt.loglog(range(1, n+1), msd_list, label="MSD about Origin")
    plt.loglog(range(1, n+1), var_list, label="MSD about Mean")
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title("Mean Square Displacement in 1D Quantum Walk")
    plt.legend()
    plt.show()

#quantum walk in 2d
def qw2d(n, coin, initState):
    
    import numpy as np
    import matplotlib.pyplot as plt

    L = 2*n + 1
    center = L//2, L//2

    psi = np.zeros((L, L, 4), dtype=complex)
    psi[center] = initState

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    msd_list = []

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

        Px = prob.sum(axis=0)
        Py = prob.sum(axis=1)

        mean_x = np.sum(x * Px)
        mean_y = np.sum(y * Py)
        msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2)
        #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py)
    
        msd_list.append(msd)

    extent = [-n, n, -n, n]
    plt.imshow(prob, origin="lower", extent=extent)
    plt.colorbar(label='Probability')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Probability of Different Positions in 2D Quantum Walk")
    plt.show()

    plt.figure()
    plt.plot(range(1, n+1), msd_list)
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title("Mean Square Displacement in 2D Quantum Walk")
    plt.show()

#quantum walk in 3d
def qw3d(n, coin, initState):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    L = 2*n + 1
    center = L//2, L//2, L//2

    psi = np.zeros((L, L, L, 8), dtype=complex)
    psi[center] = initState

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    z = np.arange(-n, n+1)
    msd_list = []

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

        Px = prob.sum(axis=(1,2))
        Py = prob.sum(axis=(0,2))
        Pz = prob.sum(axis=(0,1))

        mean_x = np.sum(x * Px)
        mean_y = np.sum(y * Py)
        mean_z = np.sum(z * Pz)
        msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2) + (np.sum(z**2 * Pz) - mean_z**2)
        #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py) + np.sum(z**2 * Pz)
    
        msd_list.append(msd)

    x, y, z = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1), np.arange(-n, n+1), indexing="ij")
    mask = prob > 1e-4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[mask], y[mask], z[mask], c=prob[mask], s=1)
    #sc = ax.scatter(x, y, z, c=prob, s=5)
    plt.colorbar(sc, ax=ax, pad=0.1, label='Probability')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Probability of Different Positions in 3D Quantum Walk")
    plt.show()

    plt.figure()
    plt.plot(range(1, n+1), msd_list)
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title("Mean Square Displacement in 3D Quantum Walk")
    plt.show()

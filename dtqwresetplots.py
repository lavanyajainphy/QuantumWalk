#quantum walk in 1d with stochastic reset to origin
def qw1dresetorigin(n, coin, initState, reset_r, n_simulations):
    
    import numpy as np
    import matplotlib.pyplot as plt

    L = 2*n + 1
    center = L//2

    x = np.arange(-n, n+1, 1)
    msd_list = np.zeros(n)
    prob_avg = np.zeros(L)

    for m in range(n_simulations):
        psi = np.zeros((L, 2, 1), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                psi = np.zeros((L, 2, 1), dtype=complex)
                psi[center] = initState
                print(f"Reset at {t} step for {m} walker")

            else:
                for i in range(L):
                        psi[i] = np.dot(coin, psi[i])

                psi_prev = psi.copy()
                psi[:] = 0
                psi[1:, 0, 0] = psi_prev [:-1, 0, 0]
                psi[:-1, 1, 0] = psi_prev [1:, 1, 0]

            prob = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
            prob_avg += prob

            mean = np.sum(x * prob)
            msd = np.sum(x**2 * prob) - mean**2
            #msd = np.sum(x**2 * prob)
            msd_list [t] += msd

    msd_list /= n_simulations

    plt.plot(x, prob_avg)
    plt.xlabel("position")
    plt.ylabel("probability")
    plt.title(f"Probability of Different Positions for {n_simulations} walkers in 1D Quantum Walk with resetting at origin")
    plt.show()

    plt.figure()
    plt.plot(range(1, n+1), msd_list)
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title(f"Mean Square Displacement of {n_simulations} walkers in 1D Quantum Walk with resetting at origin")
    plt.show()

#quantum walk in 2d with stochastic reset to origin
def qw2dresetorigin(n, coin, initState, reset_r, n_simulations):
    
    import numpy as np
    import matplotlib.pyplot as plt

    L = 2*n + 1
    center = L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    prob_avg = np.zeros((L, L))

    for m in range(n_simulations):
        psi = np.zeros((L, L, 4), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                psi = np.zeros((L, L, 4), dtype=complex)
                psi[center] = initState
                print(f"Reset at {t} step for {m} walker")

            else:
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
            prob_avg += prob
            Px = prob.sum(axis=0)
            Py = prob.sum(axis=1)

            mean_x = np.sum(x * Px)
            mean_y = np.sum(y * Py)
            msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2)
            #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py)
            msd_list [t] += msd

    msd_list /= n_simulations

    extent = [-n, n, -n, n]
    plt.imshow(prob_avg, origin="lower", extent=extent)
    plt.colorbar(label='Probability')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Probability of Different Positions for {n_simulations} walkers in 2D Quantum Walk with resetting at origin")
    plt.show()

    plt.figure()
    plt.plot(range(1, n+1), msd_list)
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title(f"Mean Square Displacement of {n_simulations} walkers in 2D Quantum Walk with resetting at origin")
    plt.show()

#quantum walk in 3d with stochastic reset to origin
def qw3dresetorigin(n, coin, initState, reset_r, n_simulations):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d

    L = 2*n + 1
    center = L//2, L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    z = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    prob_avg = np.zeros((L, L, L))

    for m in range(n_simulations):
        psi = np.zeros((L, L, L, 8), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                psi = np.zeros((L, L, L, 8), dtype=complex)
                psi[center] = initState
                print(f"Reset at {t} step for {m} walker")

            else:
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
                prob_avg += prob
                Px = prob.sum(axis=(1,2))
                Py = prob.sum(axis=(0,2))
                Pz = prob.sum(axis=(0,1))

                mean_x = np.sum(x * Px)
                mean_y = np.sum(y * Py)
                mean_z = np.sum(z * Pz)
                msd = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2) + (np.sum(z**2 * Pz) - mean_z**2)
                #msd = np.sum(x**2 * Px) + np.sum(y**2 * Py) + np.sum(z**2 * Pz)
                msd_list [t] += msd

    msd_list /= n_simulations

    x, y, z = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1), np.arange(-n, n+1), indexing="ij")
    mask = prob_avg > 1e-4
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x[mask], y[mask], z[mask], c=prob_avg[mask], s=1)
    #sc = ax.scatter(x, y, z, c=prob, s=5)
    plt.colorbar(sc, ax=ax, pad=0.1, label='Probability')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title(f"Probability of Different Positions for {n_simulations} walkers in 3D Quantum Walk with resetting at origin")
    plt.show()

    plt.figure()
    plt.plot(range(1, n+1), msd_list)
    plt.xlabel("number of steps")
    plt.ylabel("mean square displacement")
    plt.title(f"Mean Square Displacement of {n_simulations} walkers in 3D Quantum Walk with resetting at origin")
    plt.show()
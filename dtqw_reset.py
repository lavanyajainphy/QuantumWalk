#quantum walk in 1d with stochastic reset to origin
def qw1dresetorigin(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([[1], [0]])
    
    L = 2*n + 1
    center = L//2

    x = np.arange(-n, n+1, 1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
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
            mean = np.sum(x * prob)
            var = np.sum(x**2 * prob) - mean**2
            msd = np.sum(x**2 * prob)
            msd_list [t] += msd
            var_list [t] += var

        prob_final = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk1dreset_n{n}_nsim{n_simulations}_origin.npz"

    np.savez(
        filename,
        x=x,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

#quantum walk in 1d with stochastic reset to most probable position (mpp)
def qw1dresetmpp(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([[1], [0]])
    
    L = 2*n + 1
    center = L//2

    x = np.arange(-n, n+1, 1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
    prob_avg = np.zeros(L)

    for m in range(n_simulations):
        psi = np.zeros((L, 2, 1), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                prob = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
                maxima = np.where(prob == prob.max())[0]
                mpp_idx = np.random.choice(maxima)
                mpp=x[mpp_idx]
                psi = np.zeros((L, 2, 1), dtype=complex)
                psi[mpp_idx] = initState
                print(f"Reset at {t} step for {m} walker at position {mpp}")

            else:
                for i in range(L):
                        psi[i] = np.dot(coin, psi[i])

                psi_prev = psi.copy()
                psi[:] = 0
                psi[1:, 0, 0] = psi_prev [:-1, 0, 0]
                psi[:-1, 1, 0] = psi_prev [1:, 1, 0]

            prob = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
            mean = np.sum(x * prob)
            var = np.sum(x**2 * prob) - mean**2
            msd = np.sum(x**2 * prob)
            msd_list [t] += msd
            var_list [t] += var

        prob_final = np.abs(psi[:,0,0])**2 + np.abs(psi[:,1,0])**2
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk1dreset_n{n}_nsim{n_simulations}_mpp.npz"

    np.savez(
        filename,
        x=x,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

#quantum walk in 2d with stochastic reset to origin
def qw2dresetorigin(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([1, 0, 0, 0])
    
    L = 2*n + 1
    center = L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
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
            Px = prob.sum(axis=1)
            Py = prob.sum(axis=0)

            mean_x = np.sum(x * Px)
            mean_y = np.sum(y * Py)
            var = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2)
            msd = np.sum(x**2 * Px) + np.sum(y**2 * Py)
            var_list [t] += var
            msd_list [t] += msd
            
        prob_final = np.sum(np.abs(psi)**2, axis = 2)
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk2dreset_n{n}_nsim{n_simulations}_origin.npz"

    np.savez(
        filename,
        x=x,
        y=y,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

#quantum walk in 2d with stochastic reset to most probable position (mpp)
def qw2dresetmpp(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([1, 0, 0, 0])
    
    L = 2*n + 1
    center = L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
    prob_avg = np.zeros((L, L))

    for m in range(n_simulations):
        psi = np.zeros((L, L, 4), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                prob = np.sum(np.abs(psi)**2, axis = 2)
                xs, ys = np.where(prob == prob.max())
                k = np.random.randint(len(xs))
                i = xs[k]
                j = ys[k]
                psi = np.zeros((L, L, 4), dtype=complex)
                psi[i, j] = initState
                print(f"Reset at {t} step for {m} walker at ({x[i]}, {y[j]})")

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
            Px = prob.sum(axis=1)
            Py = prob.sum(axis=0)

            mean_x = np.sum(x * Px)
            mean_y = np.sum(y * Py)
            var = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2)
            msd = np.sum(x**2 * Px) + np.sum(y**2 * Py)
            var_list [t] += var
            msd_list [t] += msd
            
        prob_final = np.sum(np.abs(psi)**2, axis = 2)
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk2dreset_n{n}_nsim{n_simulations}_mpp.npz"

    np.savez(
        filename,
        x=x,
        y=y,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

#quantum walk in 3d with stochastic reset to origin
def qw3dresetorigin(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    
    L = 2*n + 1
    center = L//2, L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    z = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
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
            Px = prob.sum(axis=(1,2))
            Py = prob.sum(axis=(0,2))
            Pz = prob.sum(axis=(0,1))

            mean_x = np.sum(x * Px)
            mean_y = np.sum(y * Py)
            mean_z = np.sum(z * Pz)
            var = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2) + (np.sum(z**2 * Pz) - mean_z**2)
            msd = np.sum(x**2 * Px) + np.sum(y**2 * Py) + np.sum(z**2 * Pz)
            var_list [t] += var
            msd_list [t] += msd
            
        prob_final = np.sum(np.abs(psi)**2, axis = 3)
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")
    
    filename = f"qwalk3dreset_n{n}_nsim{n_simulations}_origin.npz"

    np.savez(
        filename,
        x=x,
        y=y,
        z=z,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

#quantum walk in 3d with stochastic reset to most probable position (mpp)
def qw3dresetmpp(n, coin, reset_r, n_simulations, initState=None):
    
    import numpy as np
    import time

    start_time = time.time()

    if initState is None:
        initState = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    
    L = 2*n + 1
    center = L//2, L//2, L//2

    x = np.arange(-n, n+1)
    y = np.arange(-n, n+1)
    z = np.arange(-n, n+1)
    msd_list = np.zeros(n)
    var_list = np.zeros(n)
    prob_avg = np.zeros((L, L, L))

    for m in range(n_simulations):
        psi = np.zeros((L, L, L, 8), dtype=complex)
        psi[center] = initState

        for t in range(n):

            if np.random.random() < reset_r:
                prob = np.sum(np.abs(psi)**2, axis = 3)
                xs, ys, zs = np.where(prob == prob.max())
                max = np.random.randint(len(xs))
                i = xs[max]
                j = ys[max]
                k = zs[max]
                psi = np.zeros((L, L, L, 8), dtype=complex)
                psi[i, j, k] = initState
                print(f"Reset at {t} step for {m} walker at ({x[i]}, {y[j]}, {z[k]})")

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
            Px = prob.sum(axis=(1,2))
            Py = prob.sum(axis=(0,2))
            Pz = prob.sum(axis=(0,1))

            mean_x = np.sum(x * Px)
            mean_y = np.sum(y * Py)
            mean_z = np.sum(z * Pz)
            var = (np.sum(x**2 * Px) - mean_x**2) + (np.sum(y**2 * Py) - mean_y**2) + (np.sum(z**2 * Pz) - mean_z**2)
            msd = np.sum(x**2 * Px) + np.sum(y**2 * Py) + np.sum(z**2 * Pz)
            var_list [t] += var
            msd_list [t] += msd
            
        prob_final = np.sum(np.abs(psi)**2, axis = 3)
        prob_avg += prob_final

    prob_avg /= n_simulations
    msd_list /= n_simulations
    var_list /= n_simulations

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")
    
    filename = f"qwalk3dreset_n{n}_nsim{n_simulations}_mpp.npz"

    np.savez(
        filename,
        x=x,
        y=y,
        z=z,
        prob=np.array(prob_avg),
        msd=np.array(msd_list),
        var=np.array(var_list),
        runtime=runtime
    )

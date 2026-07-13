def qw2d(n, L, marked, tau, reset=None, r=None, radius=None, coin=None):

    import numpy as np
    import time

    start_time = time.time()

    if coin is None:
        coin = 0.5*np.array([
            [-1,  1,  1,  1],
            [ 1, -1,  1,  1],
            [ 1,  1, -1,  1],
            [ 1,  1,  1, -1]
        ])

    center = L//2, L//2
    marked_idx = [(center[0] + y, center[1] + x) for x, y in marked]

    #marked_coin = -np.identity(4)
    marked_coin = -coin

    num_measurements = n // tau
    Fn = np.zeros(num_measurements)

    psi = np.ones((L, L, 4), dtype=complex)
    psi /= np.linalg.norm(psi)

    attempt = 0
    attempts_since_reset = 0

    survival_prob = 1.0

    for t in range(n):
      
        psi_prev = psi.copy()
        psi = psi_prev @ coin.T
        for m in marked_idx:
            psi[m] = psi_prev[m] @ marked_coin.T

        psi_prev = psi.copy()
        psi[:] = 0
      
        #w flipping of coin state, periodic boundary conditions
        psi[:, :, 1] = np.roll(psi_prev[:, :, 0], 1, axis=0) #up up spin state, shifts up, changes coin state to down up
        psi[:, :, 0] = np.roll(psi_prev[:, :, 1], -1, axis=0) #down up spin state, shifts down, changes coin state to up up
        psi[:, :, 3] = np.roll(psi_prev[:, :, 2], 1, axis=1) #up down spin state, shifts right, changes coin state to down down
        psi[:, :, 2] = np.roll(psi_prev[:, :, 3], -1, axis=1) #down down spin state, shifts left, changes coin state to up down

        if (t + 1) % tau == 0:

            attempt += 1
            attempts_since_reset += 1

            prob = np.sum(np.abs(psi)**2, axis=2)

            marked_probs = np.array([prob[m] for m in marked_idx])

            total_marked_prob = marked_probs.sum()

            #deterministic detection
            Fn[attempt - 1] = (survival_prob * total_marked_prob)
            survival_prob *= (1 - total_marked_prob)

            for m in marked_idx:
                psi[m] = 0
            psi /= np.linalg.norm(psi)

            if reset == "Uniform Superposition":
                reset_type = "UniSup"
                if attempts_since_reset == r:
                    psi = np.ones((L, L, 4), dtype=complex)
                    psi /= np.linalg.norm(psi)
                    attempts_since_reset = 0

            if reset == "Localised Superposition around Most Probable Position":
                reset_type = "LocalSup_MPP"
                if attempts_since_reset == r:
                    prob = np.sum(np.abs(psi)**2, axis = 2)
                    xs, ys = np.where(prob == prob.max())
                    i = xs[0]
                    j = ys[0]
                    psi[:] = 0
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            x = (i + di) % L
                            y = (j + dj) % L
                            psi[x, y, :] = 1
                    psi /= np.linalg.norm(psi)
                    attempts_since_reset = 0

    Pdet = np.cumsum(Fn)

    if reset is None:
        reset_type = "NA"

    if r is None:
        r = "NA"

    if radius is None:
        radius = "NA"

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk2dsearch_n{n}_reset{reset_type}_r{r}_radius{radius}.npz"

    np.savez(
        filename,
        n=n, 
        reset_type=reset_type,
        r=r,
        tau=tau,
        radius=radius,
        Fn=Fn,
        Pdet=Pdet,
        runtime=runtime
    )  

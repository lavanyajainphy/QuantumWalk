def qw2d(n, L, n_simulations, marked, tau, coin=None, reset=None, r=None, radius=None):
  
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    start_time = time.time()

    if coin is None:
        coin = 0.5*np.array([[-1,  1,  1,  1],
                             [ 1, -1,  1,  1],
                             [ 1,  1, -1,  1],
                             [ 1,  1,  1, -1]])
      
    center = L//2, L//2
    marked_idx = [(center[0] + y, center[1] + x) for x, y in marked]
    marked_coin = -np.identity(4)

    num_measurements = n // tau
    Fn = np.zeros(num_measurements)

    for _ in range(n_simulations):
        psi = np.ones((L, L, 4), dtype=complex) #begin w an equal superposition
        psi /= np.linalg.norm(psi)
        attempt = 0
        attempts_since_reset = 0
        detected = False

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

            prob = np.sum(np.abs(psi)**2, axis=2)

            if (t + 1) % tau == 0:
                attempt += 1
                attempts_since_reset += 1

                for m in marked_idx:

                    node_prob = prob[m]

                    if np.random.rand() < node_prob:
                        print(f"Found marked node {m} at {t+1}")
                        hitting_time = attempt * tau
                        Fn [attempt-1] += 1
                        detected = True
                        break
            
                    else:
                        psi[m] = 0
                        psi /= np.linalg.norm(psi)  

                if detected:
                    break  

                if reset == "Uniform Superposition":
                    reset_type = "UniSup"
                    if attempts_since_reset == r: 
                        psi = np.ones((L, L, 4), dtype=complex) #begin w an equal superposition
                        psi /= np.linalg.norm(psi)
                        attempts_since_reset = 0

                if reset == "Projective Measurement":
                    reset_type = "PM"
                    if attempts_since_reset == r: 
                        position_prob = np.sum(np.abs(psi)**2, axis=2)
                        flat_prob = position_prob.ravel()
                        flat_prob /= np.sum(flat_prob)
                        measured_idx = np.random.choice(flat_prob.size, p=flat_prob)
                        x, y = np.unravel_index(measured_idx, position_prob.shape)
                        coin_state = psi[x, y].copy()
                        psi[:] = 0
                        psi[x, y] = coin_state
                        psi /= np.linalg.norm(psi)
                        attempts_since_reset = 0

                if reset == "Localised Superposition around Projective Measurement":
                    reset_type = "LocalSup_PM"
                    if attempts_since_reset == r: 
                        position_prob = np.sum(np.abs(psi)**2, axis=2)
                        flat_prob = position_prob.ravel()
                        flat_prob /= np.sum(flat_prob)
                        measured_idx = np.random.choice(flat_prob.size, p=flat_prob)
                        i, j = np.unravel_index(measured_idx, position_prob.shape)
                        psi[:] = 0
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                x = (i + di) % L
                                y = (j + dj) % L
                                psi[x, y, :] = 1
                        psi /= np.linalg.norm(psi)
                        attempts_since_reset = 0

                if reset == "Most Probable Position":
                    reset_type = "MPP"
                    if attempts_since_reset == r: 
                        prob = np.sum(np.abs(psi)**2, axis = 2)
                        xs, ys = np.where(prob == prob.max())
                        k = np.random.randint(len(xs))
                        i = xs[k]
                        j = ys[k]
                        coin_state = psi[i, j].copy()
                        psi[:] = 0
                        psi[i, j] = coin_state
                        psi /= np.linalg.norm(psi)
                        attempts_since_reset = 0

                if reset == "Localised Superposition around Most Probable Position":
                    reset_type = "LocalSup_MPP"
                    if attempts_since_reset == r: 
                        prob = np.sum(np.abs(psi)**2, axis = 2)
                        xs, ys = np.where(prob == prob.max())
                        k = np.random.randint(len(xs))
                        i = xs[k]
                        j = ys[k]
                        psi[:] = 0
                        radius = 10
                        for di in range(-radius, radius + 1):
                            for dj in range(-radius, radius + 1):
                                x = (i + di) % L
                                y = (j + dj) % L
                                psi[x, y, :] = 1
                        psi /= np.linalg.norm(psi)
                        attempts_since_reset = 0

    Fn /= n_simulations
    Pdet = np.cumsum(Fn)

    if reset is None:
        reset_type = "NA"

    if r is None:
        r = "NA"

    if radius is None:
        radius = "NA"

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"qwalk2dsearch_n{n}_nsim{n_simulations}_reset{reset_type}_r{r}_radius{radius}.npz"

    np.savez(
        filename,
        n=n, 
        n_simulations=n_simulations,
        reset_type=reset_type,
        r=r,
        tau=tau,
        radius=radius,
        Fn=Fn,
        Pdet=Pdet,
        runtime=runtime
    )  

def rw2d(n, L, n_simulations, marked, tau, reset=None, r=None):
  
    import numpy as np
    import random
    import time

    start_time = time.time()
      
    center = L//2, L//2
    marked_idx = [(center[0] + j, center[1] + i) for i, j in marked]

    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))

    x[:, 0] = center[1]
    y[:, 0] = center[0]

    num_measurements = (n-1) // tau
    Fn = np.zeros(num_measurements)

    for i in range(n_simulations):
        attempt = 0
        attempts_since_reset = 0
        for j in range (1, n):
            direction = random.randint(1, 4)
            if direction == 1:
                x[i, j]= x[i, j-1]+1
                y[i, j]= y[i, j-1]
            elif direction == 2:
                x[i, j]= x[i, j-1]-1
                y[i, j]= y[i, j-1]
            elif direction == 3:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]+1
            else:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]-1
            x[i, j] %= L
            y[i, j] %= L

            if j % tau == 0:
                attempt += 1
                attempts_since_reset += 1
                current_position = (y[i, j], x[i, j])

                if current_position in marked_idx:
                    Fn[attempt - 1] += 1
                    break
            
                if reset == "Origin":
                    reset_type = "Origin"
                    if attempts_since_reset == r:
                        x[i, j] = center[1]
                        y[i, j] = center[0]
                        attempts_since_reset = 0

    Fn /= n_simulations
    Pdet = np.cumsum(Fn)

    if reset is None:
        reset_type = "NA"

    if r is None:
        r = "NA"

    marked_str = "_".join(f"{i}-{j}" for i, j in marked)

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

    filename = f"rwalk2dsearch_n{n}_L{L}_reset{reset_type}_r{r}_marked{marked_str}.npz"

    np.savez(
        filename,
        n=n,
        L=L,
        n_simulations = n_simulations,
        marked=marked,
        reset_type=reset_type,
        r=r,
        tau=tau,
        Fn=Fn,
        Pdet=Pdet,
        runtime=runtime
    )

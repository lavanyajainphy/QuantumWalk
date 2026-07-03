def qw2dgraph(n, coin, marked, L):
  
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    start_time = time.time()
  
    center = L//2, L//2
    marked_idx = [(center[0] + y, center[1] + x) for x, y in marked]

    psi = np.ones((L, L, 4), dtype=complex) #begin w an equal superposition
    psi /= np.linalg.norm(psi)
    marked_coin = -coin
    
    success_prob = []

    for _ in range(n):

        psi_prev = psi.copy()
        psi = psi_prev @ coin.T
        for m in marked_idx:
            psi[m] = psi_prev[m] @ marked_coin.T

        psi_prev = psi.copy()
        psi[:] = 0
        #w flipping of coin state, periodic boundary conditions
        psi[:, :, 1] = np.roll(psi_prev[:, :, 0], 1, axis=0) #up
        psi[:, :, 0] = np.roll(psi_prev[:, :, 1], -1, axis=0) #down
        psi[:, :, 3] = np.roll(psi_prev[:, :, 2], 1, axis=1) #left
        psi[:, :, 2] = np.roll(psi_prev[:, :, 3], -1, axis=1) #right

        prob = np.sum(np.abs(psi)**2, axis = 2)
        success_prob.append([prob[m] for m in marked_idx])

    imax_idx = np.unravel_index(np.argmax(prob), prob.shape)
    imax_xy = (imax_idx[1] - center[1],
             imax_idx[0] - center[0])
    print("Maximum (x,y):", imax_xy)
    print("P(max) =", prob.max())
    print("Marked (x,y):", marked)
    for coord, idx in zip(marked, marked_idx):
        print(f"P{coord} = {prob[idx]}")
    print(f"P(marked) = {sum(prob[idx] for idx in marked_idx)}")

    success_prob = np.array(success_prob)
    plt.figure()
    for i, m in enumerate(marked):
        plt.plot(success_prob[:, i], label=f"{m}")
    plt.xlabel("Number of Steps")
    plt.ylabel("Probability of Marked Point(s)")
    plt.title("Probability of finding the walker at the Marked Node(s)")
    plt.legend()
    plt.show()

    x_coords = np.arange(L) - center[1]
    y_coords = np.arange(L) - center[0]

    dx = x_coords[1] - x_coords[0]  # =1 here, but written generally
    dy = y_coords[1] - y_coords[0]

    extent = [x_coords[0] - dx/2, x_coords[-1] + dx/2,
              y_coords[0] - dy/2, y_coords[-1] + dy/2]

    im = plt.imshow(prob, origin="lower", extent=extent, vmin=0, vmax=prob.max())
    for x, y in marked:
        plt.scatter(x, y, color="red", marker="x", s=100)
    plt.colorbar(im, label='Probability')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Probability of Different Positions in 2D Quantum Walk after {n} steps")
    plt.show()

    runtime = time.time() - start_time
    print(f"The runtime is {runtime}")

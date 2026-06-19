#random walk in 1d with stochastic resetting
def rw1dreset(n, reset_r):

    import numpy as np
    import random

    x = np.zeros(n)
    n_simulations = 1000
    position = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                x[i]= 0
                step = 0
                position[i,j] = 0
                continue
            direction = random.randint(1, 2)
            if direction == 1:
                x[i]= x[i-1]+1
                step = 1
            else:
                x[i]= x[i-1]-1
                step = -1
            position[i, j] = position[i, j-1] + step

    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean(position[:, i]**2)

    filename = f"rwalk1dreset_n{n}.npz"

    np.savez(
        filename,
        x=x,
        msd=np.array(msd)
    )

#random walk in 2d with stochastic resetting
def rw2dreset(n, reset_r):

    import numpy as np
    import random

    n_simulations = 1000
    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                x[i, j]= 0
                continue
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

    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean(x[:, i]**2 + y[:, i]**2)

    filename = f"rwalk2dreset_n{n}.npz"

    np.savez(
        filename,
        x=x,
        msd=np.array(msd)
    )

#random walk in 3d with stochastic resetting
def rw3dreset(n, reset_r):
    import numpy as np
    import random

    n_simulations = 1000
    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    z = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                x[i, j]= 0
                continue
            direction = random.randint(1, 6)
            if direction == 1:
                x[i, j]= x[i, j-1]+1
                y[i, j]= y[i, j-1]
                z[i, j]= z[i, j-1]
            elif direction == 2:
                x[i, j]= x[i, j-1]-1
                y[i, j]= y[i, j-1]
                z[i, j]= z[i, j-1]
            elif direction == 3:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]+1
                z[i, j]= z[i, j-1]
            elif direction == 4:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]-1
                z[i, j]= z[i, j-1]
            elif direction == 5:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]
                z[i, j]= z[i, j-1]+1
            else:
                x[i, j]= x[i, j-1]
                y[i, j]= y[i, j-1]
                z[i, j]= z[i, j-1]-1

    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean(x[:, i]**2 + y[:, i]**2 + z[:, i]**2)

    filename = f"rwalk3dreset_n{n}.npz"

    np.savez(
        filename,
        x=x,
        msd=np.array(msd)
    )
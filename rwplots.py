#random walk in 1d
def rw1d(n, n_simulations):
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    position = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            direction = random.randint(1, 2)
            if direction == 1:
                step = 1
            else:
                step = -1
            position[i, j] = position[i, j-1] + step

    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean(position[:, i]**2)
    plt.plot(msd)
    plt.title('Mean Square Displacement of 1000 Walkers in 1D Random Walk')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

#random walk in 2d
def rw2d(n, n_simulations):
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    for i in range(n_simulations):
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

    msd = np.zeros(n)
    for i in range(n):
        msd[i] = np.mean(x[:, i]**2 + y[:, i]**2)
    plt.plot(msd)
    plt.title('Mean Square Displacement of 1000 Walkers in 2D Random Walk')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

#random walk in 3d
def rw3d(n, n_simulations):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import random

    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    z = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
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
    plt.plot(msd)
    plt.title('Mean Square Displacement of 1000 Walkers in 3D Random Walk')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

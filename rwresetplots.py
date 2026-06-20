#random walk in 1d with stochastic resetting
def rw1dreset(n, reset_r):

    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    n_simulations = 1000
    position = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                step = 0
                position[i,j] = 0
                continue
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
    plt.title('Mean Square Displacement of 1000 Walkers in 1D Random Walk with Resetting')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

#random walk in 2d with stochastic resetting
def rw2dreset(n, reset_r):

    import numpy as np
    import matplotlib.pyplot as plt
    import random

    n_simulations = 1000
    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                x[i, j]= 0
                y[i, j]= 0
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
    plt.plot(msd)
    plt.title('Mean Square Displacement of 1000 Walkers in 2D Random Walk with Resetting')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

#random walk in 3d with stochastic resetting
def rw3dreset(n, reset_r):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    import random

    n_simulations = 1000
    x = np.zeros((n_simulations, n))
    y = np.zeros((n_simulations, n))
    z = np.zeros((n_simulations, n))
    for i in range(n_simulations):
        for j in range (1, n):
            if np.random.random() < reset_r:
                x[i, j]= 0
                y[i, j]= 0
                z[i, j]= 0
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
    plt.plot(msd)
    plt.title('Mean Square Displacement of 1000 Walkers in 3D Random Walk with Resetting')
    plt.xlabel('number of steps')
    plt.ylabel('mean square displacement')
    plt.show()

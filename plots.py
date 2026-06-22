#run plt.show() after running these functions

def plot_msd(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        msd = np.load(file)["msd"]
        plt.loglog(msd, label=label)

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement")
    plt.title(title)
    plt.legend()

def plot_var(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        var = np.load(file)["var"]
        plt.loglog(var, label=label)

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement")
    plt.title(title)
    plt.legend()

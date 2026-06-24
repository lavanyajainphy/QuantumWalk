#run plt.show() after running these functions

def plot_msd(files, labels, cutoffs, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label, cutoff in zip(files, labels, cutoffs):
        msd = np.load(file)["msd"]
        steps = np.arange(1, len(msd) + 1)
        mask = steps <= cutoff
        alpha, logA = np.polyfit(np.log10(steps[mask]), np.log10(msd[mask]), 1)
        A=10**logA
        plt.loglog(steps, msd, label=f"{label}, ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement")
    plt.title(title)
    plt.legend()

def plot_var(files, labels, cutoffs, title):

    import numpy as np
    import matplotlib.pyplot as plt


    for file, label, cutoff in zip(files, labels, cutoffs):
        var = np.load(file)["var"]
        steps = np.arange(1, len(var) + 1)
        mask = steps <= cutoff
        alpha, logA = np.polyfit(np.log10(steps[mask]), np.log10(var[mask]), 1)
        A=10**logA
        plt.loglog(steps, var, label=f"{label}, ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement")
    plt.title(title)
    plt.legend()

#run plt.show() after running these functions

def plot_msd(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        msd = np.load(file)["msd"]
        steps = np.arange(1, len(msd) + 1)
        alpha, logA = np.polyfit(np.log10(steps), np.log10(msd), 1)
        A=10**logA
        plt.loglog(steps, msd, label=f"{label} ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement about Origin")
    plt.title(title)
    plt.legend()

def plot_msd_masked(files, labels, cutoff, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label, cutoff in zip(files, labels, cutoffs):
        msd = np.load(file)["msd"]
        steps = np.arange(1, len(msd) + 1)
        start, end = cutoff
        mask = (steps >= start) & (steps <= end)
        alpha, logA = np.polyfit(np.log10(steps[mask]), np.log10(msd[mask]), 1)
        A=10**logA
        plt.loglog(steps, msd, label=f"{label} ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement about Origin")
    plt.title(title)
    plt.legend()

def plot_var(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt


    for file, label in zip(files, labels):
        var = np.load(file)["var"]
        steps = np.arange(1, len(var) + 1)
        alpha, logA = np.polyfit(np.log10(steps), np.log10(var), 1)
        A=10**logA
        plt.loglog(steps, var, label=f"{label} ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement about Mean")
    plt.title(title)
    plt.legend()

def plot_var_masked(files, labels, cutoff, title):

    import numpy as np
    import matplotlib.pyplot as plt


    for file, label in zip(files, labels):
        var = np.load(file)["var"]
        steps = np.arange(1, len(var) + 1)
        start, end = cutoff
        mask = (steps >= start) & (steps <= end)
        alpha, logA = np.polyfit(np.log10(steps[mask]), np.log10(var[mask]), 1)
        A=10**logA
        plt.loglog(steps, var, label=f"{label} ($A={A:.3g}$, $\\alpha={alpha:.3f}$)")

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement about Mean")
    plt.title(title)
    plt.legend()

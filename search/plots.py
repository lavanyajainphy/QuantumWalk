def plot_pdet(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        pdet = np.load(file)["pdet"]
        n = np.load(file)["n"]
        steps = np.arange(1, n + 1)
        plt.plot(steps, pdet, label=f"{label}")

    plt.xlabel("Time step")
    plt.ylabel("Cumulative detection probability")
    plt.title(title)
    plt.legend()

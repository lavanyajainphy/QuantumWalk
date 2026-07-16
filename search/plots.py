def plot_pdet(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        pdet = np.load(file)["Pdet"]
        n = np.load(file)["n"]
        tau = np.load(file)["tau"]
        time_steps = np.arange(tau, n + 1, tau)
        plt.plot(time_steps, pdet, label=f"{label}")

    plt.xlabel("Time step")
    plt.ylabel("Cumulative detection probability")
    plt.title(title)
    plt.legend()

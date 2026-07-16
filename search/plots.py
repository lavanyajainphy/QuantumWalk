def plot_Pdet(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    for file, label in zip(files, labels):
        Pdet = np.load(file)["Pdet"]
        n = np.load(file)["n"]
        tau = np.load(file)["tau"]
        time_steps = np.arange(tau, n + 1, tau)
        plt.plot(time_steps, Pdet, label=f"{label}")

    plt.xlabel("Time step")
    plt.ylabel("Cumulative detection probability")
    plt.title(title)
    plt.legend()

def plot_radii(files, title, last_measurement, n):

    import numpy as np
    import matplotlib.pyplot as plt
    all_radius = []
    all_Pdet = []

    for file in files:
        Pdet = np.load(file)["Pdet"][last_measurement]
        all_Pdet.append(Pdet)
        radius = np.load(file)["radius"]
        all_radius.append(radius)

    plt.xlabel("Radius")
    plt.ylabel(f"Pdet at {n}th Step")
    plt.scatter(all_radius, all_Pdet, color="black")
    plt.plot(all_radius, all_Pdet, color="black")
    plt.title(title)

def plot_msd(files, labels, title):

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()

    for file, label in zip(files, labels):
        msd = np.load(file)["msd"]
        steps = np.arange(1, len(msd) + 1)
        plt.loglog(steps, msd, label=label)

    plt.xlabel("Number of steps")
    plt.ylabel("Mean square displacement")
    plt.title(title)
    plt.legend()
    plt.show()
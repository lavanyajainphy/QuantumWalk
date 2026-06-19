#for hadamard, theta = np.pi/4, phi1=phi2=0
#for grover, theta = np.pi/2, phi1=phi2=0
#for fourier, theta = np.pi/4, phi1=phi2=np.pi/2

def coin1d(theta, phi1, phi2):
    import numpy as np
    return np.array([[np.cos(theta), (np.sin(theta) * np.exp(1j*phi1))], [(np.sin(theta) * np.exp(1j*phi2)), -(np.cos(theta) * np.exp(1j*(phi1 + phi2)))]])

def coin2d(theta, phi1, phi2):
    import numpy as np
    c1 = np.array([[np.cos(theta), (np.sin(theta) * np.exp(1j*phi1))], [(np.sin(theta) * np.exp(1j*phi2)), -(np.cos(theta) * np.exp(1j*(phi1 + phi2)))]])
    return np.kron(c1, c1)

def coin3d(theta, phi1, phi2):
    import numpy as np
    c1 = np.array([[np.cos(theta), (np.sin(theta) * np.exp(1j*phi1))], [(np.sin(theta) * np.exp(1j*phi2)), -(np.cos(theta) * np.exp(1j*(phi1 + phi2)))]])
    c2 = np.kron(c1, c1)
    return np.kron(c2, c1)
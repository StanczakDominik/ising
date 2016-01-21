import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt

random.seed(1)

N=4
J=1
k=1
T=1
beta = 1/k/T

spins=np.ones([N,N], int)
spins[:,1::2] = -1

def E(spins):
    total = 0
    center = spins[1:-1, 1:-1]
    sides = spins[:-2, 1:-1] + spins[2:, 1:-1] +\
        spins[1:-1, 2:] + spins[1:-1, :-2]

    total += np.sum(center*sides)

    upper_left = spins[0,0] * (spins[1,0] + spins[0,1])
    lower_left = spins[0,-1] * (spins[1, -1] + spins[0,-2])
    upper_right = spins[-1, 0] * (spins[-2,0] + spins[-1,1])
    lower_right = spins[-1,-1] * (spins[-1,-2] + spins[-2,-1])
    total += upper_left + lower_left + upper_right + lower_right

    return -J*total

def M(spins):
    return np.sum(spins)

print(E(spins), M(spins))
print(spins)

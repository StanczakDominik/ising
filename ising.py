import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.pyplot as plt
from time import time

# random.seed(1)
def ising(N, NT, T, plotting=False, show=False):

    start_time = time()

    Noffset=N+2
    k=1
    J=1
    beta=1/k/T
    max_e = J*N**2
    max_m = N**2
    NT = int(NT)
    saved_parameters=3
    history = np.empty((NT+1,saved_parameters))

    def Energy(spins):
        center = spins[1:-1, 1:-1]
        sides = spins[:-2, 1:-1] + spins[2:, 1:-1] +\
            spins[1:-1, 2:] + spins[1:-1, :-2]

        return -J*np.sum(center*sides)

    def Mag(spins):
        return np.sum(spins[1:-1,1:-1])

    def FlipSpin(spins, E, M):
        x, y = random.randint(1,N+2, 2)
        test = spins[x,y]
        deltaE=J*2*test
        accepted = 0
        if(random.random() < np.exp(-beta*deltaE)):
            E += deltaE
            M -= 2*test
            spins[x,y] *= -1
            accepted = 1
        return E, M, accepted
    def ViewSystem(title):
        print(title)
        print("Energy: %d\tMagnetization: %d\tN: %d\tT: %.2f" %(E,M,N,T))
        print(spins[1:-1])

    spins=np.ones([Noffset,Noffset], int)
    spins[:,0] = spins[:,-1] = spins[0,:] = spins[-1, :] = 0
    # spins[1:-1,1:-1:2,] = -1
    spins[1:-1, 1:-1] = random.randint(0,2, (N,N))*2-1


    E, M = Energy(spins), Mag(spins)
    parameters = E, M, -1
    history[0] = parameters
    ViewSystem("Starting")

    for i in range(NT):
        parameters = E, M, Accepted = FlipSpin(spins,E,M)
        history[i+1]= parameters
    ViewSystem("Finished")
    energies = history[:,0]
    magnetization = history[:,1]
    acceptance = history[:,2]
    print("Acceptance ratio: %f" %np.mean(acceptance))
    print("Runtime: %f" % (time()-start_time))

    def plot():
        fig, (ax_energy, ax_magnet) = plt.subplots(2, sharex=True, sharey=False,figsize=(15,7) )

        ax_energy.plot(range(NT+1),energies, "b-", label="Energy")
        ax_energy.plot(range(NT+1), np.ones(NT+1)*max_e, "b--", label="Max energy")
        ax_energy.plot(range(NT+1), -np.ones(NT+1)*max_e, "b--")
        # ax_energy.legend()
        ax_energy.set_ylabel("Energy")

        ax_magnet.plot(range(NT+1),magnetization, "g-", label="Magnetization")
        ax_magnet.plot(range(NT+1), np.ones(NT+1)*max_m, "g--", label="Max magnetization")
        ax_magnet.plot(range(NT+1), -np.ones(NT+1)*max_m, "g--")
        ax_magnet.set_ylabel("Magnetization")
        # ax_magnet.legend()


        plt.title("Energy: %d Magnetization: %d N: %d T: %.2f" %(E,M,N,T))
        plt.xlabel("Time")


        title_string = "N%d_T%.2f.png"%(N,T)
        plt.savefig(title_string)
        if(show):
            plt.show()
        else:
            plt.clf()
    if(plotting):
        plot()

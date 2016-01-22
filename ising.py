import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from time import time

# random.seed(1)
def ising(N, NT, T, plotting=False, show=False, anim=False, restart=True):

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
    if anim:
        Nsnapshots = int(np.sqrt(NT))
        snapshot_history = np.empty((Nsnapshots, N,N), int)

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
        deltaE=-J*2*test
        accepted = 0
        if(random.random() < np.exp(-beta*deltaE)):
            E += deltaE
            M -= 2*test
            accepted = test
            spins[x,y] *= -1
        return E, M, accepted
    def ViewSystem(title):
        print(title)
        print("Energy: %d\tMagnetization: %d\tN: %d\tT: %.2f" %(E,M,N,T))
        print(spins[1:-1,1:-1])

    def ShowSpins(spins):
        plot = plt.imshow(spins)
        plot.set_cmap('Greys_r')
        plt.colorbar()
        plt.show()

    spins=np.ones([Noffset,Noffset], int)
    spins[:,0] = spins[:,-1] = spins[0,:] = spins[-1, :] = 0
    # spins[1:-1,1:-1:2,] = -1
    if restart:
        spins[1:-1, 1:-1] = random.randint(0,2, (N,N))*2-1
        np.save("data/N%d_T%.2f_start"%(N,T), spins)
    else:
        spins=np.load("N%d_T%.2f_start.npy"%(N,T))
    E, M = Energy(spins), Mag(spins)
    parameters = E, M, -1
    history[0] = parameters
    ViewSystem("Starting")

    for i in range(NT):
        parameters = E, M, Accepted = FlipSpin(spins,E,M)
        history[i+1]= parameters
        if anim and i%(Nsnapshots)==0:
            snapshot_history[int((i/NT)*Nsnapshots)]=spins[1:-1, 1:-1]

    np.save("data/N%d_T%.2f_finish"%(N,T), spins)
    ViewSystem("Finished")
    energies = history[:,0]
    magnetization = history[:,1]
    acceptance = history[:,2]
    print("Average spin from which transition was accepted: %f" %np.mean(acceptance))
    print("Runtime: %f" % (time()-start_time))

    def plot():
        fig, (ax_energy, ax_magnet, ax_acceptance) = plt.subplots(3, sharex=True, sharey=False,figsize=(15,12) )
        plt.title("Energy: %d Magnetization: %d N: %d T: %.2f" %(E,M,N,T))

        ax_energy.plot(range(NT+1),energies, "b-", label="Energy")
        ax_energy.plot(range(NT+1), np.ones(NT+1)*max_e, "b--", label="Max energy")
        ax_energy.plot(range(NT+1), -np.ones(NT+1)*max_e, "b--")
        # ax_energy.legend()
        ax_energy.set_ylabel("Energy")

        ax_magnet.plot(range(NT+1),magnetization, "g-", label="Magnetization")
        ax_magnet.plot(range(NT+1), np.ones(NT+1)*max_m, "g--", label="Max magnetization")
        ax_magnet.plot(range(NT+1), -np.ones(NT+1)*max_m, "g--")
        ax_magnet.set_ylabel("Magnetization")

        ax_acceptance.plot(range(NT+1),acceptance, "g-", label="Magnetization")
        ax_acceptance.set_ylabel("Accepted change from spin:")
        # ax_magnet.legend()


        plt.xlabel("Time")


        title_string = "plots/N%d_T%.2f.png"%(N,T)
        plt.savefig(title_string)
        if(show):
            plt.show()
        else:
            plt.clf()
    if(plotting):
        plot()

    def animate():
        fig=plt.figure()
        plot = plt.imshow(snapshot_history[0], interpolation="nearest")
        plot.set_cmap('Greys_r')
        plt.colorbar()
        title = plt.title("Iteration: %d"%0)

        def update_fig(i):
            title = plt.title("i: %d M: %d E: %d"%(i, magnetization[i], energies[i]))
            plot.set_array(snapshot_history[i//Nsnapshots])
            return plot, title

        ani = matplotlib.animation.FuncAnimation(fig, update_fig, np.arange(0,NT-1,NT//Nsnapshots), interval=30, blit=False, repeat=True)
        ani.save('plots/cabbage.mp4', writer='mencoder', fps=30)
        plt.show()
    if(anim):
        animate()
ising(2,5,2, plotting=True)#, show=True, anim=True)

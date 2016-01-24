import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from time import time
import os.path
k=1
J=1
Theory_TC = J/k*2/np.log(1+np.sqrt(2))

def ising(N, NT, T, plotting=False, show=False, anim=False, continue_run=True):

    beta=1/k/T
    NT = int(NT)
    saved_parameters=4
    Nsnapshots = int(NT**(1/3))
    if(T<Theory_TC):
        Theory_M = N**2*(1-np.sinh(2*beta*J)**(-4))**(1/8)

    parameter_string = "data/N%d_T%.1f/"%(N,T)

    if not os.path.exists(r"./" + parameter_string):
        continue_run = False
        os.makedirs(r"./" + parameter_string)
        spins = random.randint(0,2, (N,N))*2-1
        np.save(parameter_string+"data_start", spins)
        history = np.zeros((Nsnapshots,saved_parameters))
        starting_iteration = starting_history_iteration = 0
    elif(continue_run):
        spins=np.load(parameter_string+"data_finish.npy")
        history=np.load(parameter_string+"history.npy")
        starting_iteration = int(history[-1,0]+1)
        starting_history_iteration = history.shape[0]
        history=np.append(history,np.zeros((Nsnapshots,saved_parameters)), axis=0)
    else:
        spins=np.load(parameter_string+"data_start.npy")
        history = np.zeros((Nsnapshots,saved_parameters))
        starting_iteration = starting_history_iteration = 0

    if anim:
        snapshot_history = np.empty((Nsnapshots, N,N), int)
    start_time = time()

    def Energy(spins):
        center = spins
        sides = np.roll(spins, 1, 0) + np.roll(spins, 1, 1) + np.roll(spins, -1, 0) + np.roll(spins, -1, 1)
        return -J*np.sum(center*sides)/2

    def Mag(spins):
        return np.sum(spins)

    def FlipSpin(spins, E, M):
        x, y = random.randint(0,N,2)
        test = spins[x,y]
        neighbor_spins = spins[(x+1)%N,(y)%N] + spins[(x-1)%N,y%N] + spins[x%N,(y-1)%N] + spins[x%N,(y+1)%N]
        deltaE=J*2*test*neighbor_spins
        accepted = 0
        uniform_random = random.random()
        probability_cutoff = np.exp(-beta*deltaE)
        if(uniform_random < probability_cutoff):
            E += deltaE
            M -= 2*test
            accepted = 1
            spins[x,y] *= -1
        return E, M, accepted
    def ViewSystem(title):
        print(title)
        print("Iteration: %d\tEnergy: %d\tMagnetization: %d\tN: %d\tT: %.2f" %(starting_iteration,E,M,N,T))
        print(spins[1:-1,1:-1])

    E, M = Energy(spins), Mag(spins)
    ViewSystem("Starting")
    for i in range(NT):
        E, M, Accepted = FlipSpin(spins,E,M)
        if i%(Nsnapshots)==0:
            snapshot_iteration = int((i/NT)*Nsnapshots)
            parameters = i+starting_iteration, E, M, Accepted
            history[starting_history_iteration+snapshot_iteration]= parameters
            if anim:
                snapshot_history[snapshot_iteration]=spins

    np.save(parameter_string+"data_finish", spins)
    np.save(parameter_string+"history", history)
    ViewSystem("Finished")


    times = history[:,0]
    energies = history[:,1]
    magnetization = history[:,2]
    acceptance = history[:,3]
    print(np.mean(acceptance))
    print("Acceptance ratio: %f" %np.mean(acceptance))
    print("Runtime: %f" % (time()-start_time))

    def plot():
        fig, (ax_energy, ax_magnet) = plt.subplots(2, sharex=True, sharey=False,figsize=(15,7) )
        plt.title("Final energy: %d Final magnetization: %d N: %d T: %.2f" %(E,M,N,T))

        ax_energy.plot(times,energies, "b-", label="Energy")
        # ax_energy.plot(times, np.ones_like(times)*max_e, "b--", label="Max energy")
        # ax_energy.plot(times, -np.ones_like(times)*max_e, "b--")
        ax_energy.legend()
        ax_magnet.grid()
        ax_energy.set_ylabel("Energy")

        ax_magnet.plot(times,magnetization, "g-", label="Magnetization")
        if(T<Theory_TC):
            ax_magnet.plot(times, np.ones_like(times)*Theory_M, "g--", label="Theoretical spontaneous")
            ax_magnet.plot(times, -np.ones_like(times)*Theory_M, "g--")
        ax_magnet.set_ylabel("Magnetization")
        ax_magnet.legend()
        ax_magnet.grid()
        plt.xlabel("Time")

        plt.savefig(parameter_string+"plot.png")
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
            title = plt.title("i: %d M: %d E: %d"%(i*NT/Nsnapshots+starting_iteration, magnetization[i], energies[i]))
            plot.set_array(snapshot_history[i])
            return plot, title

        ani = matplotlib.animation.FuncAnimation(fig, update_fig, np.arange(0,Nsnapshots), interval=30, blit=False, repeat=True)
        if(show):
            plt.show()
        else:
            plt.clf()
    if(anim):
        animate()

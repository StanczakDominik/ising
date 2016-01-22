import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from time import time
k=1
J=1
Theory_TC = J/k*2/np.log(1+np.sqrt(2))

def ising(N, NT, T, plotting=False, show=False, anim=False, restart=True, continue_run=False):

    start_time = time()


    beta=1/k/T
    NT = int(NT)
    saved_parameters=3
    Nsnapshots = int(NT**(1/3))
    history = np.empty((Nsnapshots,saved_parameters))
    if anim:
        snapshot_history = np.empty((Nsnapshots, N,N), int)

    Theory_M = N**2*(1-np.sinh(2*beta*J)**(-4))**(1/8)
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
        print("Energy: %d\tMagnetization: %d\tN: %d\tT: %.2f" %(E,M,N,T))
        print(spins[1:-1,1:-1])

    if restart:
        spins = random.randint(0,2, (N,N))*2-1
        np.save("pbcdata/N%d_T%.2f_start"%(N,T), spins)
    elif continue_run:
        spins=np.load("pbcdata/N%d_T%.2f_finish.npy"%(N,T))
    else:
        spins=np.load("pbcdata/N%d_T%.2f_start.npy"%(N,T))

    E, M = Energy(spins), Mag(spins)
    parameters = E, M, -1
    history[0] = parameters
    ViewSystem("Starting")

    for i in range(NT):
        E, M, Accepted = FlipSpin(spins,E,M)
        if anim and i%(Nsnapshots)==0:
            parameters = E, M, Accepted
            history[int((i/NT)*Nsnapshots)]= parameters
            snapshot_history[int((i/NT)*Nsnapshots)]=spins

    np.save("pbcdata/N%d_T%.2f_finish"%(N,T), spins)
    ViewSystem("Finished")
    energies = history[:,0]
    magnetization = history[:,1]
    acceptance = history[:,2]
    print("Acceptance ratio: %f" %np.mean(acceptance))
    print("Runtime: %f" % (time()-start_time))

    def plot():
        fig, (ax_energy, ax_magnet) = plt.subplots(2, sharex=True, sharey=False,figsize=(15,7) )
        plt.title("Energy: %d Magnetization: %d N: %d T: %.2f" %(E,M,N,T))

        times = np.linspace(0,NT,Nsnapshots)
        ax_energy.plot(times,energies, "b-", label="Energy")
        # ax_energy.plot(times, np.ones(Nsnapshots)*max_e, "b--", label="Max energy")
        # ax_energy.plot(times, -np.ones(Nsnapshots)*max_e, "b--")
        ax_energy.legend()
        ax_magnet.grid()
        ax_energy.set_ylabel("Energy")

        ax_magnet.plot(times,magnetization, "g-", label="Magnetization")
        ax_magnet.plot(times, np.ones(Nsnapshots)*Theory_M, "g--", label="Theoretical spontaneous")
        ax_magnet.plot(times, -np.ones(Nsnapshots)*Theory_M, "g--")
        ax_magnet.set_ylabel("Magnetization")
        ax_magnet.legend()
        ax_magnet.grid()
        plt.xlabel("Time")


        title_string = "pbcplots/N%d_T%.2f.png"%(N,T)
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
            title = plt.title("i: %d M: %d E: %d"%(i*NT/Nsnapshots, magnetization[i], energies[i]))
            plot.set_array(snapshot_history[i])
            return plot, title

        ani = matplotlib.animation.FuncAnimation(fig, update_fig, np.arange(0,Nsnapshots), interval=30, blit=False, repeat=True)
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('pbcplots/cabbage.mp4', writer='ffmpeg', fps=30)
        if(show):
            plt.show()
        else:
            plt.clf()
    if(anim):
        animate()
ising(256,1e6,0.1, plotting=True, show=True, anim=True, restart=False, continue_run=True)

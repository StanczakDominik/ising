import numpy as np
import numpy.random as random
import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt
from scipy.special import ellipk
from time import time
import os.path


k=1             #Boltzmann's constant
J=1             #the exchange integral
Theory_TC = J/k*2/np.log(1+np.sqrt(2)) #the theoretical value for the critical temperature

def Theory_U_Formula(T):
    """
    Calculates the theoretical energy per site as seen over at
    https://en.wikipedia.org/wiki/Square-lattice_Ising_model#Exact_solution
    T: Temperature, in units of Boltzmann's constant.
    """
    beta = 1/k/T
    k_parameter = 1/np.sinh(2*beta*J)**2
    m = 4*k_parameter*(1+k_parameter)**(-2)
    integral = ellipk(m)

    return -J/np.tanh(2*beta*J)*(1+2/np.pi*(2*np.tanh(2*beta*J)**2-1)*integral)

def ising(N, NT, T, plotting=False, show=False, anim=False, continue_run=True):
    """
    Runs a 2D square lattice Ising model simulation using periodic boundary conditions.

    N: number of rows of the square lattice. N=64 gives 64^2 spins in the system.
    NT: number of time steps, or single flip trials. Preferably as a float:
        1e6 will run for a million iteration. Please limit this to 1e8 at most.
    T:  Temperature, in units of Boltzmann's constant (this can be set to its
        actual physical value above).
    plotting: set to True to get a history plot of energy and magnetization
        for the whole simulation.
    anim: set to True to get an animation of the current run. NT^(1/3) snapshots
        are taken to conserve memory and speed up the simulation.
    continue_run: set to False to restart the run from its starting data.
        Note that this will overwrite any data already there beside starting
        from the same initial condition.
    """

    ##=============Setup parameters =======================
    beta=1/k/T
    NT = int(NT)
    saved_parameters=4  #how many parameters do we want to keep in history
    Nsnapshots = int(NT**(1/3)) #how many spin array snapshots to take. ^1/3 because it seemed okay.
    Theory_U = Theory_U_Formula(T)*N**2     #theoretical total energy for the simulation area
    if(T<Theory_TC): #there is no spontaneous magnetization otherwise
        Theory_M = N**2*(1-np.sinh(2*beta*J)**(-4))**(1/8) #see wikipedia link above for formula


    ##================File management, history loading, etc.=================
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


    ##========= Define useful functions and initial diagnostics==========
    def Energy(spins):
        """
        Takes the array of spins and calculates its energy (with PB conditions)
        """
        center = spins
        sides = np.roll(spins, 1, 0) + np.roll(spins, 1, 1) + np.roll(spins, -1, 0) + np.roll(spins, -1, 1)
        return -J*np.sum(center*sides)/2

    def Mag(spins):
        """
        Takes the array of spins and calculates its total magnetization.
        Doesn't get much simpler than this.
        """
        return np.sum(spins)

    def FlipSpin(spins, E, M):
        """
        The core of the Metropolis-Hastings algorithm.
        Operates off relative changes in energy and magnetization to save time.

        1. pick a spin at random
        2. calculate the sum of spins of its 4 neighbors
        3. calculate the change in energy of the system should the random spin flip
        4. calculate the Boltzmann probability factor - how likely is the spin to flip?
            for T>0 this is between 0 to 1.
        5. pick a float from 0 to 1 at random
        6. if the random float is lesser than the Boltzmann cutoff, flip the spin
           (in-place) and update the energies.
        """

        x, y = random.randint(0,N,2)    #1.
        test = spins[x,y]
        neighbor_spins = spins[(x+1)%N,(y)%N] + spins[(x-1)%N,y%N] + spins[x%N,(y-1)%N] + spins[x%N,(y+1)%N] #2.
        deltaE=J*2*test*neighbor_spins #3.
        accepted = 0
        probability_cutoff = np.exp(-beta*deltaE) #4.
        uniform_random = random.random() #5.
        if(uniform_random < probability_cutoff): #6.
            E += deltaE
            M -= 2*test
            accepted = 1
            spins[x,y] *= -1
        return E, M, accepted

    def ViewSystem(title):
        """
        Print the state of the system to console.
        """
        print(title)
        print("Iteration: %d\tEnergy: %d\tMagnetization: %d\tN: %d\tT: %.2f" %(starting_iteration,E,M,N,T))
        print(spins[1:-1,1:-1])

    ## ==================== Main loop ===================
    start_time = time()     #start timing the run

    E, M = Energy(spins), Mag(spins)    #first (and only) direct calculation
    ViewSystem("Starting")
    for i in range(NT):
        E, M, Accepted = FlipSpin(spins,E,M)
        if i%(Nsnapshots)==0:    #saves data to history
            snapshot_iteration = int((i/NT)*Nsnapshots)
            parameters = i+starting_iteration, E, M, Accepted
            history[starting_history_iteration+snapshot_iteration]= parameters
            if anim:
                snapshot_history[snapshot_iteration]=spins

    #saves the final spin array
    np.save(parameter_string+"data_finish", spins)
    #saves the history array, for the energy and magnetization plot
    np.save(parameter_string+"history", history)
    ViewSystem("Finished")



    #diagnostics
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
        ax_energy.plot(times, np.ones_like(times)*Theory_U, "r--", label="Theoretical energy")
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
        ax = fig.add_subplot(111)

        ims = [[ax.imshow(snapshot_history[i], interpolation='nearest', animated=True, cmap='Greys_r'),\
         plt.title("T: %f i: %d M: %d E: %d"%(T,times[i], magnetization[i], energies[i]))] for i in np.arange(0,Nsnapshots)]
        ani = matplotlib.animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
        ani.save(parameter_string+"video%d.mp4"%starting_iteration, fps=30, extra_args=['-vcodec', 'libx264'])
        if(show):
            plt.show()
        else:
            plt.clf()
    if(anim):
        animate()

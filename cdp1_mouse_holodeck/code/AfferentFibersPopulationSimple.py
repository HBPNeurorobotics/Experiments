import pyNN.nest as sim
import numpy as np
import scipy.stats as stats
from pyNN.utility import Timer
from CellsParameters import CellsParameters


class AfferentFibersPopulationSimple(sim.Population):
    """ Afferent fiber population model
    It can be a target of a stimulation model
    and you can set the instantaneous natural firing rates.
    However, no antidromic collsions are modelled; for this
    purpose use the AfferentFibersPopulation model
    """

    __maxFiringRate = 250
    __minSynapticDelay = .21 # Due to the implementation of population.set->spike_times. less than 2.1 don't work. Need to check
    __stimWeight = 4

    def __init__(self,nCells,delay=2,label=None):
        self._nCells = nCells
        self._label = label
        self._delay = delay

        sim.Population.__init__(self,nCells,CellsParameters.afferentFiber(),label=self._label)

        self._sensoryInfo = sim.Population(nCells,sim.SpikeSourceArray(),label='sensoryInformation')
        syn = sim.StaticSynapse(weight=self.__class__.__stimWeight,delay=self._delay)
        self._connections = sim.Projection(self._sensoryInfo, self, sim.OneToOneConnector(), syn, receptor_type='excitatory')
        self.initialize_cells_activity()

    def initialize_cells_activity(self,lastSpikeTime=-9999):
        self._spikeTimes = []
        self._timeLastPulses = [lastSpikeTime]*self._nCells

    def set_firing_rate(self,firingRate,pulsesNumber=20):
        """ Set the afferent firing rate.

        Keyword arguments:
        firingRate -- firing rate in Hz
        """

        # Recover the time when the last pulse occurred
        time = sim.get_current_time()

        if self._spikeTimes:
            for i,spikeTimes in enumerate(self._spikeTimes):
                if np.any(spikeTimes<time):
                    self._timeLastPulses[i] = spikeTimes[spikeTimes<time].max()

        # Compute the new pulse times
        if firingRate>0:

            if firingRate>=self.__class__.__maxFiringRate:
                # print "Warning: the firingRate exceeds the predefined maximum value (%d)."%(firingRate)
                firingRate=self.__class__.__maxFiringRate

            lower = 1000.0/self.__class__.__maxFiringRate
            upper = 99999.
            mu = 1000.0/firingRate #ms
            sigma = mu*0.2
            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            intervals = x.rvs(self._nCells)

            self._spikeTimes = []
            for timeLastPulse,interval in zip(self._timeLastPulses,intervals):
                firstPulseTime = timeLastPulse+interval if timeLastPulse+interval>time+self.__class__.__minSynapticDelay \
                    else time-np.random.uniform(interval/2.,interval-self.__class__.__minSynapticDelay,1)
                self._spikeTimes.append(np.arange(firstPulseTime,firstPulseTime+pulsesNumber*interval,interval))

        elif firingRate<=0:
            self._spikeTimes = np.array([])


        self._sensoryInfo.set(spike_times=self._spikeTimes)

    def get_delay(self):
        """ Return the time delay in ms needed by a spike to travel the whole fiber. """
        return self._delay

    @classmethod
    def get_stim_weight(cls):
        """ Return the weight of a connection between an ees object and this cell. """
        return AfferentFibersPopulationSimple.__stimWeight

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Init parameters
    nCells = 100
    tstop = 1000

    sim.setup(timestep=0.1, min_delay=0.1, max_delay=3.0)

    # Create AfferentFibersPopulation instance
    af = AfferentFibersPopulationSimple(nCells)
    af.set_firing_rate(100)


    # Create target population
    parameters = {'cm': 0.25,
                 'i_offset': 0.0,
                 'tau_m': 0.000030,
                 'tau_refrac': 10,
                 'tau_syn_E': 0.5,
                 'tau_syn_I': 8.,
                 'v_reset': -65.0,
                 'v_rest': -65.0,
                 'v_thresh': -50.0}
    interneurons = sim.Population(1,sim.IF_curr_alpha(**parameters),label='stimTarget')
    syn = sim.StaticSynapse(weight=25*100,delay=0.1)
    random = sim.FixedProbabilityConnector(p_connect=1)
    connections = sim.Projection(af, interneurons, random, syn, receptor_type='excitatory')
    interneurons.record('v')
    af.record('v')

    # Running the simulation
    dt = 20
    while sim.get_current_time()<tstop:
        sim.run(dt)
        if sim.get_current_time()%100 < sim.get_time_step()/2:

            print sim.get_current_time(),int(sim.get_current_time()/10.-10)
            af.set_firing_rate(int(sim.get_current_time()/10.-10))

    # Extracting the data
    MembranePot = interneurons.get_data()
    # MembranePot = af.get_data()

    fig, ax = plt.subplots(1, 1, figsize=(16,4))
    for array in MembranePot.segments[0].analogsignalarrays:
        for i in range(array.shape[1]):
            ax.plot(array.times, array[:, i])

    plt.show()

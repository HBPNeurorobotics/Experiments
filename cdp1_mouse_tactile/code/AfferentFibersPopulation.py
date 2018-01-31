import pyNN.nest as sim
import numpy as np
import scipy.stats as stats
from pyNN.utility import Timer

class AfferentFibersPopulation(sim.Population):

    __minDelay = 1
    __maxDelay = 100
    __updatePeriod = 0.1
    __maxEesFrequency = 1001
    __maxFiringRate = 200 # This should be lower than the frequency allowed by the refractory period
    __synapticDelay =  .21 # Due to the implementation of population.set->spike_times. less than 2.1 don't work. Need to check

    def __init__(self,nCells,delay,label,debug=False):
        self._nCells = nCells
        self._label = label
        self._debug = debug
        self._set_delay(delay)
        self._maxSensorySpikesXtime = int(float(self._delay)/1000.*float(self.__class__.__maxFiringRate)+2)
        self._maxEesSpikesXtime = int(float(self._delay)/1000.*float(self.__class__.__maxEesFrequency)+2)

        # Initialize stimulation parameters
        self._recruitedFiber = 0
        self._stimPulsesTime = np.array([])

        # Set refractory period of mean: 1.6 ms - 625 Hz
        np.random.seed(1234)
        lower = 1
        upper = 1000./self.__class__.__maxFiringRate-0.1
        mu = 1.6 #ms
        sigma = mu*0.1
        x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        self._refractoryPeriod = x.rvs(self._nCells)

        #Position along the fiber recruited by the stimulation
        self._stimPosition = self._delay-0.5
        # Stim variables
        self._nRecruitedFibers = 0
        self._stimPulsesTime = np.array([])
        # tolerance to check for events
        self._tolerance = self.__class__.__updatePeriod/10.

        # Initialise afferents firings activity
        self.initialize_cells_activity()

        # Create a population of SpikeSourceArray that will be used to model afferents overal activity
        sim.Population.__init__(self,self._nCells,sim.SpikeSourceArray(),label=self._label)


    """
    Specific Methods of this class
    """
    def initialize_cells_activity(self,lastSpikeTime=0):
        """ Initialise the fiber. """
        #Sometimes when working with nans you can get annoying warnings (e.g. [nan,3]<0)
        np.seterr(invalid='ignore')
        #Initial firing rate of .1 Hz
        self._interval = 9999.*np.ones(self._nCells)
        self._oldFr = 0.
        self._lastSpikeTime = lastSpikeTime*np.ones(self._nCells)
        self._lastNaturalSpikeTime = lastSpikeTime*np.ones(self._nCells)
        self._lastAntiSpikeTime = lastSpikeTime*np.ones(self._nCells)
        #Create array containing the natural sensory spikes
        self._naturalSpikes = np.nan*np.ones([self._nCells, self._maxSensorySpikesXtime]) #[None]*self._maxSensorySpikesXtime
        self._naturalSpikesArrivalTime = np.nan*np.ones([self._nCells, self._maxSensorySpikesXtime]) #[None]*self._maxSensorySpikesXtime
        self._stimAntidromicSpikesArrivalTime = np.nan*np.ones([self._nCells, self._maxSensorySpikesXtime]) #[None]*self._maxSensorySpikesXtime
        #Create array containing the EES induced spikes
        self._eesSpikes = np.nan*np.ones([self._nCells, self._maxEesSpikesXtime]) #[None]*self._maxEesSpikesXtime
        #Last spike in stim position
        self._lastStimPosSpikeTime = -9999.*np.ones(self._nCells)
        #Stats
        self._nCollisions = 0
        self._nNaturalSent = 0
        self._nNaturalArrived = 0
        self._eesSent = 0
        self._eesArrived = 0
        self._nAntidromicRefractory = 0

    # The delay correspond to the value naturalSpikes[] should have before being sent
    def _set_delay(self,delay):
        """ Set the delay.

        Keyword arguments:
        delay -- time delay in ms needed by a spike to travel the whole fiber
        """

        if delay>=self.__class__.__minDelay and delay<=self.__class__.__maxDelay:
            self._delay=delay
        else: raise Exception("Afferent fiber delay out of limits")

    def set_firing_rate(self,fr,noise=True):
        """ Set the afferent firing rate.

        Keyword arguments:
        fr -- firing rate in Hz
        """

        if fr == self._oldFr: return
        if fr<=0:
            self._interval = 99999.*np.ones(self._nCells)
        elif fr>=self.__class__.__maxFiringRate:
            self._interval = (1000.0/self.__class__.__maxFiringRate)*np.ones(self._nCells)
        elif fr<self.__class__.__maxFiringRate and noise:
            lower = 1000.0/self.__class__.__maxFiringRate
            upper = 99999.
            mu = 1000.0/fr #ms
            sigma = mu*0.2
            x = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            self._interval = x.rvs(self._nCells)
        else: self._interval = (1000.0/fr)*np.ones(self._nCells) #ms
        self._oldFr = fr

    def setup_stimulation_amplitude(self,percRecruitedFibers):
        """ Set the stimulation amplitude, i.e. the number of fibers recrutied by the stimulation
        """
        if percRecruitedFibers<=1 and percRecruitedFibers>=0:
            self._nRecruitedFibers = int(percRecruitedFibers*self._nCells)
        else: raise(Exception("Invalid percentage of stimulated afferents."))

    def setup_stimulation_timing(self,stimPulsesTime):
        """ Set the timing of the stimulation pulses
        """
        if len(stimPulsesTime)>1 and 1000./np.min(np.diff(stimPulsesTime))>self.__class__.__maxEesFrequency:
            raise(Exception("The stimulation frequency is higher than the limit imposed by the AfferentFibersPopulation model."))
        elif len(stimPulsesTime)>0 and np.min(stimPulsesTime) <= sim.get_current_time():
            raise(Exception("The first stimulation pulse need to occur at a future time point."))
        else: self._stimPulsesTime = np.array(stimPulsesTime)

    def update(self):
        """ Update function that hase to be called every __updatePeriod ms of simulation.

        The fucntions:
            1)  propagates the action pontentials (APs) induced by the stimulation along
                the fiber
            2)  checks whether a new pulse of stimulation occured and in this case sends an event
                to all the connected cells at the time = time
            3)  It checks whether a natural AP reached the end of the fiber and in this case
                it sends an event to the connected cells at time = time.
            4)  It propagates the natural action pontentials (APs) along the fibers
                taking in to account possible collision with EES induced AP.
        """
        time = sim.get_current_time()
        # Check whether an antidromic stimulation spike arrived at the fiber origin
        indexesFibers,indexesSpikes  = np.nonzero(self._stimAntidromicSpikesArrivalTime-time <= self.__class__.__updatePeriod/2)
        if indexesFibers.size:
            if self._debug: print "\t\t\tantidromic stim spike arrived at fibe origin at time: ",time
            self._lastAntiSpikeTime[indexesFibers] = time
            # Remove spike from pipeline
            self._stimAntidromicSpikesArrivalTime[indexesFibers,indexesSpikes] = np.nan

        # Check whether a sensory spike arrived at stimulation location
        indexesFibers,indexesSpikes  = np.nonzero(self._naturalSpikesArrivalTime-time <= self.__class__.__updatePeriod/2)
        if indexesFibers.size:
            if self._debug: print "\t\t\t\t\t\t\tsensory spike arrived at time: ",time
            self._lastStimPosSpikeTime[indexesFibers] = time
            # Send a synaptic event to the attached neurons
            self[indexesFibers].set(spike_times=[time+self._delay-self._stimPosition+self.__class__.__synapticDelay])
            self._nNaturalArrived+=indexesFibers.size
            # Remove spike from pipeline
            self._naturalSpikesArrivalTime[indexesFibers,indexesSpikes] = np.nan

        #Check whether a new pulse of stimulation occured
        if self._nRecruitedFibers>0 and len(self._stimPulsesTime)>0 and np.min(np.abs(self._stimPulsesTime-time)) < self.__class__.__updatePeriod/2:
            if self._debug: print "\t\t Stimulation pulse occurred at time: %d - %d recruited fibers" % (time,self._nRecruitedFibers)
            self._eesSent += self._nRecruitedFibers
            #check whether the fibers are't in refractory period
            indexesFibersReadyToFire = np.flatnonzero(time-self._lastStimPosSpikeTime>self._refractoryPeriod) # here all fibers
            # take only the recruited fibers
            indexesRecruitedFibers = indexesFibersReadyToFire[indexesFibersReadyToFire<=self._nRecruitedFibers-1]
            if indexesRecruitedFibers.size:
                self._lastStimPosSpikeTime[indexesRecruitedFibers] = time
                # Send a synaptic event to the attached neurons
                self[indexesRecruitedFibers].set(spike_times=[time+self._delay-self._stimPosition+self.__class__.__synapticDelay])
                self._eesArrived+=indexesRecruitedFibers.size
                # Check for antidromic collisions
                indexesAllFibersAntidromic,_ = np.nonzero(self._naturalSpikesArrivalTime-time >= self.__class__.__updatePeriod/2)
                _,singleIndFibers = np.unique(indexesAllFibersAntidromic,return_index=True)
                indexesFibersAntidromic = np.intersect1d(indexesRecruitedFibers,indexesAllFibersAntidromic[singleIndFibers])
                if indexesFibersAntidromic.size:
                    if self._debug: print "\t\t%d  antidromic collision occured at time: %d" % (indexesFibersAntidromic.size,time)
                    indCollision = np.nanargmin(self._naturalSpikesArrivalTime[indexesFibersAntidromic,:],axis=1)
                    self._naturalSpikesArrivalTime[indexesFibersAntidromic,indCollision] = np.nan
                    self._nCollisions+=indexesFibersAntidromic.size
                # Add antidromic stim spike to pipeline
                indexFibersFiring = np.setdiff1d(indexesRecruitedFibers,indexesFibersAntidromic)
                if indexFibersFiring.size:
                    fiberId,indNan = np.nonzero(np.isnan(self._stimAntidromicSpikesArrivalTime[indexFibersFiring,:]))
                    _,singleIndNanPerFiber = np.unique(fiberId,return_index=True)
                    self._stimAntidromicSpikesArrivalTime[indexFibersFiring,indNan[singleIndNanPerFiber]] = time+self._stimPosition


        # Take the fibers generating a new sensory spike
        indexesFibers = np.flatnonzero((time-self._lastNaturalSpikeTime)>=self._interval-(self.__class__.__updatePeriod/2.))
        if indexesFibers.size:
            self._nNaturalSent += indexesFibers.size
            self._lastNaturalSpikeTime[indexesFibers]=time
            if self._debug: print "\t\t\t\t\t\t%d sensory spike generated at time: %d" % (indexesFibers.size,time)
            # Check fibers refractory period due to antidromic spike generated by the stimulation
            indexesAllFibersInRefractory = np.flatnonzero(time-self._lastAntiSpikeTime<=self._refractoryPeriod)
            # Conside the fibers in refractory period as andtidromic collision
            indexesFibersInRefractory = np.intersect1d(indexesFibers,indexesAllFibersInRefractory)
            if indexesFibersInRefractory.size:
                if self._debug: print "\t\t\t\t\t\t%d fiber couldn't fire because of refractory period, time: %d" % (indexesFibersInRefractory.size,time)
                self._nCollisions+=indexesFibersInRefractory.size
            # Take the fibers ready to fire
            indexesFibersReadyToFire = np.setdiff1d(indexesFibers,indexesFibersInRefractory)
            if indexesFibersReadyToFire.size:
                # Check for antidromic collisions
                indexesAllFibersAntidromic,_ = np.nonzero(self._stimAntidromicSpikesArrivalTime-time >= self.__class__.__updatePeriod/2)
                _,singleIndFibers = np.unique(indexesAllFibersAntidromic,return_index=True)
                indexesFibersAntidromic = np.intersect1d(indexesFibersReadyToFire,indexesAllFibersAntidromic[singleIndFibers])
                if indexesFibersAntidromic.size:
                    if self._debug: print "\t\t\t\t\t\t%d  antidromic collision occured at time: %d" % (indexesFibersAntidromic.size,time)
                    indCollision = np.nanargmin(self._stimAntidromicSpikesArrivalTime[indexesFibersAntidromic,:],axis=1)
                    self._stimAntidromicSpikesArrivalTime[indexesFibersAntidromic,indCollision] = np.nan
                    self._nCollisions+=indexesFibersAntidromic.size
                # Add new sensory spike to the pipeline
                indexFibersFiring = np.setdiff1d(indexesFibersReadyToFire,indexesFibersAntidromic)
                if indexFibersFiring.size:
                    fiberId,indNan = np.nonzero(np.isnan(self._naturalSpikesArrivalTime[indexFibersFiring,:]))
                    _,singleIndNanPerFiber = np.unique(fiberId,return_index=True)
                    self._naturalSpikesArrivalTime[indexFibersFiring,indNan[singleIndNanPerFiber]] = time+self._stimPosition


    def _update_ees_deprecated(self):
        """ Update the fiber activity induced by the stimulation.

        It first propagates the action pontentials (APs) induced by the stimulation along
        the fiber and then it checks whether a new pulse of stimulation occured.
        In this case an event is sent to all the connected cells at the time = time

        Keyword arguments:
        time -- current simulation time, necessary for synaptic connections
        """

        time = sim.get_current_time()
        #Propagates the ees antidromic action pontentials
        self._eesSpikes[self._eesSpikes<0]=np.nan
        self._eesSpikes -= self.__class__.__updatePeriod
        #Check whether a new pulse of stimulation occured
        if self._nRecruitedFibers == 0: return
        if len(self._stimPulsesTime)>0 and np.min(np.abs(self._stimPulsesTime-time)) < self.__class__.__updatePeriod/2:
            self._eesSent += self._nRecruitedFibers
            #check whether the fibers are't in refractory period
            indexesFibersReadyToFire = np.nonzero(time-self._lastStimPosSpikeTime>self._refractoryPeriod) # here all fibers
            indexesRecruitedFibers = indexesFibersReadyToFire[0][indexesFibersReadyToFire[0]<=self._nRecruitedFibers-1]
            if indexesRecruitedFibers.size:
                self._lastStimPosSpikeTime[indexesRecruitedFibers] = time
                self[indexesRecruitedFibers].set(spike_times=[time+self._delay-self._stimPosition+self.__class__.__synapticDelay])
                for indRecruitedFiber in indexesRecruitedFibers:
                    try: indNan = np.nonzero(np.isnan(self._eesSpikes[indRecruitedFiber,:]))[0][0]
                    except IndexError: print "No free (NaN) space on _eesSpikes..."
                    self._eesSpikes[indRecruitedFiber,indNan] = self._stimPosition
                    self._eesArrived+=indRecruitedFiber.size

    def _update_natural_deprecated(self):
        """ Update the fiber activity induced by the sensory organ.

        It first check whether a natural AP reached the end of the fiber
        and in this case it send an event to the connected cells at time = time.
        Then it propagates the natural action pontentials (APs) along the fiber
        taking in to account possible collision with EES induced AP.

        Keyword arguments:
        time -- current simulation time, necessary for synaptic connections and
        for triggering new natural APs
        """
        #check for event
        time = sim.get_current_time()
        indFiringFribers,indSpikePos = np.nonzero(self._naturalSpikes>=self._delay-self._tolerance)
        if indFiringFribers.size:
            self[indFiringFribers].set(spike_times=[time+self.__class__.__synapticDelay])
            self._naturalSpikes[indFiringFribers,indSpikePos] = np.nan
            self._nNaturalArrived += indFiringFribers.size
        #update _naturalSpikes
        nanSpots = np.isnan(self._naturalSpikes)

        for k in xrange(self._nCells):
            for i in xrange(self._maxSensorySpikesXtime):
                if np.isnan(self._naturalSpikes[k,i]): continue
                indCollision = np.nonzero(np.logical_or(np.logical_or(np.logical_or(\
                    self._naturalSpikes[k,i] > self._eesSpikes[k,:]-self._tolerance,\
                    self._naturalSpikes[k,i] < self._eesSpikes[k,:]+self._tolerance),\
                    self._naturalSpikes[k,i]+self.__class__.__updatePeriod > self._eesSpikes[k,:]-self._tolerance),\
                    self._naturalSpikes[k,i]+self.__class__.__updatePeriod < self._eesSpikes[k,:]+self._tolerance))
                if indCollision[0].size:
                    self._naturalSpikes[k,i] = np.nan
                    self._eesSpikes[k,indCollision[0][0]] = np.nan
                    self._nCollisions+=1

        #advance natural AP
        self._naturalSpikes += self.__class__.__updatePeriod
        #check for refractory period
        indFibers,indSpikes = np.nonzero(np.logical_and(self._naturalSpikes>self._stimPosition-self._tolerance,\
            self._naturalSpikes < self._stimPosition+self._tolerance))
        if indFibers.size:
            for indFiber,indSpike in zip(indFibers,indSpikes):

                if time - self._lastStimPosSpikeTime[indFiber] <= self._refractoryPeriod[indFiber]:
                    self._naturalSpikes[indFiber,indSpike]=np.nan
                else: self._lastStimPosSpikeTime[indFiber] = time

        #check for new AP
        indexesFibersReadyToFire = np.nonzero((time-self._lastSpikeTime)>=self._interval-(self.__class__.__updatePeriod/2.))
        if indexesFibersReadyToFire[0].size:
            self._lastSpikeTime[indexesFibersReadyToFire[0]]=time
            #add new AP
            fiberId,indNan = np.nonzero(np.isnan(self._naturalSpikes[indexesFibersReadyToFire[0],:]))
            _,singleIndNanPerFiber = np.unique(fiberId,return_index=True)
            self._naturalSpikes[indexesFibersReadyToFire[0],indNan[singleIndNanPerFiber]]=0
            self._nNaturalSent+=indexesFibersReadyToFire[0].size

    def get_delay(self):
        """ Return the time delay in ms needed by a spike to travel the whole fiber. """
        return self._delay

    def get_stats(self):
        """ Return a touple containing statistics of the fiber after a simulation is performed. """
        if float(self._nNaturalArrived+self._nCollisions)==0: percErasedAp = 0
        else: percErasedAp = float(100*self._nCollisions)/float(self._nNaturalArrived+self._nCollisions)
        return self._nNaturalSent,self._nNaturalArrived,self._nCollisions,percErasedAp,self._eesSent,self._eesArrived

    @classmethod
    def get_update_period(cls):
        """ Return the time period between calls of the update fcn. """
        return AfferentFibersPopulation.__updatePeriod

    @classmethod
    def get_max_ees_frequency(cls):
        """ Return the weight of a connection between an ees object and this cell. """
        return AfferentFibersPopulation.__maxEesFrequency

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Init parameters
    nCells = 200
    delay = 10
    tstop = 250
    segmentToRecord = 1.
    dt = AfferentFibersPopulation.get_update_period()
    sim.setup(timestep=0.1, min_delay=0.1, max_delay=1.0)

    # Create AfferentFibersPopulation instance
    af = AfferentFibersPopulation(nCells,delay,'testPopulation',debug=False)
    af.set_firing_rate(70,True)

    # Set up Stimulation
    af.setup_stimulation_amplitude(1)
    af.setup_stimulation_timing(np.arange(5,500,25))
    # af.setup_stimulation_amplitude(0)
    # af.setup_stimulation_timing([])

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

    # Running the simulation
    while sim.get_current_time()<tstop:
        sim.run(dt)
        # af.update_activity()
        af.update()
        if sim.get_current_time()%10 < sim.get_time_step()/2: print sim.get_current_time()
    print af.get_stats()

    # Extracting the data
    interneuronsMembranePot = interneurons.get_data()

    fig, ax = plt.subplots(1, 1, figsize=(16,4))
    for array in interneuronsMembranePot.segments[0].analogsignalarrays:
        for i in range(array.shape[1]):
            ax.plot(array.times, array[:, i])

    plt.show()

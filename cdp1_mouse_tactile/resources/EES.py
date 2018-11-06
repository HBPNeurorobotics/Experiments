import pyNN.nest as sim
import numpy as np
from scipy import interpolate
from AfferentFibersPopulation import AfferentFibersPopulation
from AfferentFibersPopulationSimple import AfferentFibersPopulationSimple
from CellsParameters import CellsParameters

import random as rnd
import time

import os
pathToCode = os.environ['NN_CODE']


class EES():
    """ Epidural Electrical Stimulation model. """

    def __init__(self,neuralNetwork,amplitude,frequency):
        """ Object initialization.

        Keyword arguments:
        neuralNetwork -- NeuralNetwork instance to connect to this object.
        amplitude -- Aplitude of stimulation. It could either be an integer
        value between _minCur and _maxCur or a list containing the percentages
        of recruited primary afferents, secondary afferents and motoneurons.
        frequency -- Stimulation frequency in Hz; it has to be lower than the
        maximum stimulation frequency imposed by the AfferentFiber model.
        pulsesNumber -- number of pulses to send (default 100000).
        """
        self._nn = neuralNetwork

        self._maxFrequency = AfferentFibersPopulation.get_max_ees_frequency()
        self._current = None
        self._percIf= None
        self._percIIf= None
        self._percMn = None

        self._projections = {}

        self._stim = sim.Population(1,sim.SpikeSourceArray(),label="stim")
        self._connect_to_network()

        # Load the recruitment data
        self._load_rec_data()

        # lets define which type of cells are recruited by the stimulation
        self._recruitedCells = sum([self._nn.get_afferents_names(),self._nn.get_motoneurons_names()],[])
        # Set stimulation parameters
        self._pulseTimes = np.array([])
        self._sentPulses = []
        self.set_amplitude(amplitude)
        self.set_frequency(frequency)


    def _load_rec_data(self):
        """ Load recruitment data from a previosly validated FEM model (Capogrosso et al 2013). """
        recI_MG=np.loadtxt(pathToCode+'/recruitmentData/GM_full_S1_wire1')
        recII_MG=np.loadtxt(pathToCode+'/recruitmentData/GM_full_ii_S1_wire1')
        recMn_MG=np.loadtxt(pathToCode+'/recruitmentData/MGM_full_S1_wire1')
        recI_TA=np.loadtxt(pathToCode+'/recruitmentData/TA_full_S1_wire1')
        recII_TA=np.loadtxt(pathToCode+'/recruitmentData/TA_full_ii_S1_wire1')
        recMn_TA=np.loadtxt(pathToCode+'/recruitmentData/MTA_full_S1_wire1')

        allPercIf_GM= recI_MG/max(recI_MG)
        allPercIIf_GM= recII_MG/max(recII_MG)
        allPercMn_GM = recMn_MG/max(recMn_MG)
        allPercIf_TA= recI_TA/max(recI_TA)
        allPercIIf_TA= recII_TA/max(recII_TA)
        allPercMn_TA = recMn_TA/max(recMn_TA)

        self._minCur = 0 #uA
        self._maxCur = 600 #uA

        nVal = recI_MG.size
        allPercIf= (allPercIf_GM+allPercIf_TA)/2
        allPercIIf= (allPercIIf_GM+allPercIIf_TA)/2
        allPercMn = (allPercMn_GM+allPercMn_TA)/2

        currents = np.linspace(self._minCur,self._maxCur,nVal)
        self._tckIf = interpolate.splrep(currents, allPercIf)
        self._tckIIf = interpolate.splrep(currents, allPercIIf)
        self._tckMn = interpolate.splrep(currents, allPercMn)

    def _connect_to_network(self):
        """ Connect this object to the NeuralNetwork object. """
        # Iterate over all DoFs
        for muscle in self._nn.cells:
            self._projections[muscle] = {}
            # Iterate over all type of cells
            for cellName in self._nn.cellLabels[muscle]:
                if cellName in self._nn.get_motoneurons_names() or \
                    (self._nn.use_simple_afferent_models() and cellName in self._nn.get_afferents_names()):
                    conWeight = 0
                    receptorType = "excitatory"
                    delay = 0.1
                    syn = sim.StaticSynapse(weight=conWeight,delay=delay)
                    connector = sim.AllToAllConnector()
                    postsynapticCells = self._nn.cells[muscle].get_population(cellName)
                    self._projections[muscle][cellName] = sim.Projection(self._stim, postsynapticCells, connector, syn, receptor_type=receptorType)

    def _activate_connections(self,connections,percentage,wValue):
        """ Modify which connections are active. """
        nActiveConnections = int(percentage*connections.size())
        weights = np.zeros(connections.size())
        weights[:nActiveConnections]= wValue
        connections.set(weight=weights)

    def set_amplitude(self,amplitude,muscles=None):
        """ Set the amplitude of stimulation.

        Note that currently all DoFs have the same percentage of afferents recruited.
        Keyword arguments:
        amplitude -- Aplitude of stimulation. It coulde either be an integer
        value between _minCur and _maxCur or a list containing the percentages
        of recruited primary afferents, secondary afferents and motoneurons.
        muscles -- list of muscle names on which the stimulation amplitude is
        modifiel. If no value is specified, none is used and all the amplitude is
        modified on all the network muscles.
        """

        if isinstance(amplitude,int) or isinstance(amplitude,float):
            if amplitude > self._minCur and amplitude <self._maxCur:
                self._current = amplitude
                self._percIf=  interpolate.splev(amplitude,self._tckIf)
                if self._percIf<0:self._percIf=0
                self._percIIf=  interpolate.splev(amplitude,self._tckIIf)
                if self._percIIf<0:self._percIIf=0
                self._percMn =  interpolate.splev(amplitude,self._tckMn)
                if self._percMn<0:self._percMn=0
            else:
                raise Exception("Current amplitude out of bounds - min = "+str(self._minCur)+"/ max = "+str(self._maxCur))
        elif isinstance(amplitude,list) and len(amplitude)==3:
            self._current = -1
            self._percIf= amplitude[0]
            self._percIIf= amplitude[1]
            self._percMn = amplitude[2]
        else: raise Exception("badly defined amplitude")

        if muscles is None: muscles = self._nn.cells.keys()
        for muscle in muscles:
            # Iterate over all type of cells
            for cellName in self._nn.cellLabels[muscle]:
                if cellName in self._nn.get_primary_afferents_names():
                    if self._nn.use_simple_afferent_models():
                        self._activate_connections(self._projections[muscle][cellName],self._percIf,AfferentFibersPopulationSimple.get_stim_weight())
                    else:
                        self._nn.cells[muscle].get_population(cellName).setup_stimulation_amplitude(self._percIf)
                elif cellName in self._nn.get_secondary_afferents_names():
                    if self._nn.use_simple_afferent_models():
                        self._activate_connections(self._projections[muscle][cellName],self._percIIf,AfferentFibersPopulationSimple.get_stim_weight())
                    else:
                        self._nn.cells[muscle].get_population(cellName).setup_stimulation_amplitude(self._percIIf)
                elif cellName in self._nn.get_motoneurons_names():
                    self._activate_connections(self._projections[muscle][cellName],self._percMn,CellsParameters.motoneuronsStimWeight())


    def set_frequency(self,frequency,pulsesNumber=10000):
        """ Set the frequency of stimulation.

        Note that currently all DoFs have the same percentage of afferents recruited.
        Keyword arguments:
        frequency -- Stimulation frequency in Hz; it has to be lower than the
        maximum stimulation frequency imposed by the AfferentFiber model.
        """

        # Recover the time when the last pulse occurred
        time = sim.get_current_time()
        if len(self._pulseTimes)>0:
            timeLastPulse = self._pulseTimes[self._pulseTimes<time].max()
            # add last pulses to the recording list
            self._sentPulses = sum([self._sentPulses,self._pulseTimes[self._pulseTimes<time]],[])
        else: timeLastPulse = -99999.

        # Compute the new pulse times
        if frequency>0 and frequency<self._maxFrequency:
            self._frequency = frequency
            interval = 1000.0/self._frequency
            minDelay = .21
            firstPulseTime = timeLastPulse+interval if timeLastPulse+interval>time+minDelay else time+minDelay
            self._pulseTimes = np.arange(firstPulseTime,firstPulseTime+pulsesNumber*interval,interval)
        elif frequency<=0:
            self._frequency = 0
            self._pulseTimes = np.array([])
        elif frequency>=self._maxFrequency:
            raise(Exception("The stimulation frequency exceeds the maximum frequency imposed by the AfferentFiber model."))

        self._stim.set(spike_times=self._pulseTimes)
        if not self._nn.use_simple_afferent_models():
            for muscle in self._nn.cells:
                # Iterate over all type of cells
                for cellName in self._nn.cellLabels[muscle]:
                    if cellName in self._nn.get_afferents_names():
                        self._nn.cells[muscle].get_population(cellName).setup_stimulation_timing(self._pulseTimes)


    def get_amplitude(self,printFlag=False):
        """ Return the stimulation amplitude and print it to screen.

        Current bug: if set_amplitude was used with the non default 'muscles' parameter,
        the stimulation amplitude here returned is not valid for the whole network.
        Indeed, this function only returns the most recent amplitude value that was used
        to change the stimulation settings. """

        if printFlag:
            print "The stimulation amplitude is set at: "+str(self._current)+" uA"
            print "\t"+str(int(self._percIf*100))+"% of primary afferents recruited"
            print "\t"+str(int(self._percIIf*100))+"% of secondary afferents recruited"
            print "\t"+str(int(self._percMn*100))+"% of motoneuron recruited"
        return self._current, self._percIf, self._percIIf,    self._percMn

    def get_frequency(self,printFlag=False):
        """ Return the stimulation frequency and print it to screen. """
        frequency = None
        if printFlag: print "The stimulation frequency is set at: "+str(self._frequency)+" Hz"
        return self._frequency

    def get_pulses(self):
        """ Return the stimulation pulses. """
        # add last pulses to the recording list
        time = sim.get_current_time()
        if len(self._pulseTimes)>0:
            self._sentPulses = sum([self._sentPulses,self._pulseTimes[self._pulseTimes<time]],[])
        return self._sentPulses

if __name__ == '__main__':
    from SpinalNeuralNetwork import SpinalNeuralNetwork
    from AfferentFibersPopulation import AfferentFibersPopulation

    tstop = 50
    dt = AfferentFibersPopulation.get_update_period()
    sim.setup(timestep=dt/2., min_delay=dt/2., max_delay=2.0)

    nn = SpinalNeuralNetwork("frwSimRat.txt")
    stim = EES(nn,235,50)

    while sim.get_current_time()<tstop:
        sim.run(dt)
        nn.update_afferents_ap()
        if sim.get_current_time()%10 < sim.get_time_step()/2: print sim.get_current_time()

import os
import time
import pyNN.nest as sim
pathToCode = os.environ['NN_CODE']
from AfferentFibersPopulation import AfferentFibersPopulation
from SpinalNeuralNetwork import SpinalNeuralNetwork
from EES import EES
import random as rnd
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from tools import firings_tools as tlsf
from tools import afferents_tools as tlsa
#from tools import networkPlotter as nnplt
import pickle


class ClosedLoopSimulation:
    """ Closed loop simulation of a neuro-biomechanical model.

    Here the modeled neural network is integrated over time given the
    sensory information coming from the musculoskeletal model.

    Communication
        Information coming from the musculoskeletal model:
            - "time", i.e.: "6.54"
            - "muscleName stretch", i.e.: "right_ta 2.22"
            - "muscleName stretch"
            - "muscleName stretch"
            - ...
        Information to send at the musculoskeletal model:
            - "muscleName muscActivation", i.e. "right_ta 0.68"
            - "muscleName muscActivation"
            - "muscleName muscActivation"
    """

    def __init__(self, neuralNetworkStructure, species, eesAmplitude, eesFrequency, resultsName="test", dtCommunication=20):
        """ Object initialization.

        Keyword arguments:
        neuralNetwork -- NeuralNetwork object.
        species -- 'mouse', 'rat' or 'human' (default == human))
        ees -- possible EES object connected to the NeuralNetwork. If no EES objects
        are connected use None (default = None).
        tStop -- Time in ms at wich the simulation will stop (default = 999999).
        """

        self._debug = False
        self._plotFlag = False

        # Setup results folder
        self._resultsFolder = pathToCode+"/../../results/"
        if not os.path.exists(self._resultsFolder):
            os.makedirs(self._resultsFolder)

        if self._debug: print "\nSetting up simulation environment..."
        extra = {'threads' : 4}
        sim.setup(timestep=0.1, min_delay=0.1, max_delay=2.0,**extra)
        if self._debug: print "\nCreating simulation variables..."

        self._nn = SpinalNeuralNetwork(neuralNetworkStructure)
        self._ees = EES(self._nn,eesAmplitude,eesFrequency)
        self._ees.get_amplitude(True)
        self._species = species
        self._resultsName = resultsName
        self._dtCommunication = dtCommunication

        self._init_afferents_fr()
        self._init_muscles_param()

        self._tMuscModel = 0
        self.maxMnFr = 50. #Hz

        if self._plotFlag: self._initialize_plot_data()
        if not self._nn.use_simple_afferent_models():
            self._integrationStep = AfferentFibersPopulation.get_update_period()
        else: self._integrationStep = self._dtCommunication


    def run_step(self,mmData):
        time = sim.get_current_time()
        if self._debug: print "doing a step of simulation..."
        self._tLastStep = time
        self._update_mm_data(mmData)
        if self._debug: print "data received..."
        self._update_afferents_fr()
        self._nn.set_afferents_fr(self._afferentFr)
        if self._debug: print "afferents updated..."
        while time-self._tLastStep <= (self._dtCommunication-0.5*self._integrationStep):
            self._integrate()
            self._update()
            time = sim.get_current_time()
        if self._debug: print "dt integration completed."
        self._compute_musc_act()
        if self._plotFlag:
            if self._debug: print "plotting data.."
            self._plot()
        return self._muscAct

    def terminate(self):
        """ Print the total simulation time and extract the results. """
        self._extract_results()
        self._save_results()
        #if self._plotFlag: self._plotter.end_plotting()

    def _init_afferents_fr(self):
        """ Initialize the dictionary necessary to update the afferent fibers. """
        self._afferentFr = {}
        for muscleName in self._nn.cells:
            self._afferentFr[muscleName]={}
            for cellName in self._nn.cellLabels[muscleName]:
                if cellName in self._nn.get_afferents_names():
                    self._afferentFr[muscleName][cellName]= 0.

    def _init_muscles_param(self):
        """ Initialize muscles parameters dictionaries. """
        muscleName = self._nn.cells.keys()[0]

        self._nIaInt = self._nn.cells[self._nn.cells.keys()[0]].get_population("IaInt").size
        self._nIIExInt = self._nn.cells[self._nn.cells.keys()[0]].get_population("IIExInt").size
        self._nIaf = self._nn.cells[self._nn.cells.keys()[0]].get_population("Iaf").size
        self._nIIf = self._nn.cells[self._nn.cells.keys()[0]].get_population("IIf").size
        self._nMn = self._nn.cells[muscleName].get_population("Mn").size

        self._muscStretch = {}
        self._muscStretchOld = {}
        self._muscStretchVel = {}
        self._muscAct = {}
        self._nApMn = {}
        self._nApIaInt = {}
        self._nApIIExInt = {}
        self._nApIaf = {}
        self._nApIIf = {}

        self._timeWindowRec = 250.# ms
        nRecordingWindows = np.ceil(self._timeWindowRec/self._dtCommunication)

        for muscle in self._nn.cells:
            self._muscStretch[muscle] = 0.
            self._muscStretchOld[muscle] = 0.
            self._muscStretchVel[muscle] = 0.
            self._muscAct[muscle] = 0.
            self._nApMn[muscle] = np.zeros([int(nRecordingWindows)])
            self._nApIaInt[muscle] = np.zeros([int(nRecordingWindows)])
            self._nApIIExInt[muscle] = np.zeros([int(nRecordingWindows)])
            self._nApIaf[muscle] = np.zeros([int(nRecordingWindows)])
            self._nApIIf[muscle] = np.zeros([int(nRecordingWindows)])

    def _initialize_plot_data(self):
        self._cmap = plt.get_cmap('viridis')
        self._maxIaIntFr = 150.
        self._maxIIExIntFr = 150.
        self._maxIafFr = 150.
        self._maxIIfFr = 150.
        movie = {'folder':self._resultsFolder,
                'name':self._resultsName,
                'fps':int(1000./self._dtCommunication)}
        movie = None # ffmpeg problems in linux machine..
        #self._plotter = nnplt.musclespindles_network_plotter("LEFT_LG","LEFT_TA",movieParam=movie,rosPublish=True)

    def _update_mm_data(self,mmData):
        """ Read and update muscles stretch in mm and time in ms form the musculoskeletal model. """
        # update values of the previous step
        for muscle in self._muscStretch: self._muscStretchOld[muscle]=self._muscStretch[muscle]
        tMuscModelOld = self._tMuscModel

        # Read the current time in the musculoskeletal model.
        self._tMuscModel = mmData["t"]
        dtMuscModel = self._tMuscModel-tMuscModelOld # this dt should be equal to the closedLoop period of the neural networ

        # Read and update the muscles stretch from the musculoskeletal model.
        for muscle in mmData['stretch']:
            # The string coming from the musculoskeletal model has to be formatted as: "muscle(str) stretch(float) in m"
            self._muscStretch[muscle] = mmData['stretch'][muscle]*1000 # transform it in mm

        # Update the muscles stretch velocity
        if sim.get_current_time() != 0:
            for muscle in self._muscStretch:
                self._muscStretchVel[muscle] = (self._muscStretch[muscle]-self._muscStretchOld[muscle])/dtMuscModel*1000 # dt is in ms

    def _update_afferents_fr(self):
        """ Estimate and update the afferents firing rate."""
        for muscle in self._afferentFr:
            # compute the afferents fr
            self._afferentFr[muscle]['Iaf'] = tlsa.compute_Ia_fr(self._muscStretch[muscle],self._muscStretchVel[muscle],self._muscAct[muscle],self._species)
            self._afferentFr[muscle]['IIf'] = tlsa.compute_II_fr(self._muscStretch[muscle],self._muscAct[muscle],self._species)

    def _integrate(self):
        """ Integrate the neuronal cells for a defined integration time step ."""
        sim.run(self._integrationStep)

    def _update(self):
        """ Update simulation parameters. """
        if not self._nn.use_simple_afferent_models(): self._nn.update_afferents_ap()

    def _compute_musc_act(self):
        """ Compute the muscle activations """
        nApMnNew = self._nn.get_ap_number(["Mn"])
        # Update the recorded windows of nMnAp
        for muscle in self._nApMn:
            for i in xrange(self._nApMn[muscle].size-1):
                self._nApMn[muscle][i] = self._nApMn[muscle][i+1]
            self._nApMn[muscle][-1] = nApMnNew[muscle]["Mn"]
            mnFr = (self._nApMn[muscle][-1]-self._nApMn[muscle][0])/self._nMn/self._timeWindowRec*1000
            self._muscAct[muscle] = mnFr/self.maxMnFr
            if self._muscAct[muscle]>1: self._muscAct[muscle]=1

    def _plot(self):

        nApIaIntNew = self._nn.get_ap_number(["IaInt"])
        nApIIExIntNew = self._nn.get_ap_number(["IIExInt"])
        nApIafNew = self._nn.get_ap_number(["Iaf"])
        nApIIfNew = self._nn.get_ap_number(["IIf"])

        IaIntFr={}
        IIExIntFr={}
        IIfFr={}
        IafFr={}

        activityColors = {}
        firingRates = {}
        # Update the recorded windows
        for muscle in self._nn.cells:
            for i in xrange(self._nApIaInt[muscle].size-1):
                self._nApIaInt[muscle][i] = self._nApIaInt[muscle][i+1]
            for i in xrange(self._nApIIExInt[muscle].size-1):
                self._nApIIExInt[muscle][i] = self._nApIIExInt[muscle][i+1]
            for i in xrange(self._nApIaf[muscle].size-1):
                self._nApIaf[muscle][i] = self._nApIaf[muscle][i+1]
            for i in xrange(self._nApIIf[muscle].size-1):
                self._nApIIf[muscle][i] = self._nApIIf[muscle][i+1]

            self._nApIaInt[muscle][-1] = nApIaIntNew[muscle]["IaInt"]
            self._nApIIExInt[muscle][-1] = nApIIExIntNew[muscle]["IIExInt"]
            self._nApIaf[muscle][-1] = nApIafNew[muscle]["Iaf"]
            self._nApIIf[muscle][-1] = nApIIfNew[muscle]["IIf"]

            firingRates[muscle] = {}
            firingRates[muscle]["IaInt"] = (self._nApIaInt[muscle][-1]-self._nApIaInt[muscle][0])/self._nIaInt/self._timeWindowRec*1000
            firingRates[muscle]["IIExInt"] = (self._nApIIExInt[muscle][-1]-self._nApIIExInt[muscle][0])/self._nIIExInt/self._timeWindowRec*1000
            firingRates[muscle]["Iaf"] = (self._nApIaf[muscle][-1]-self._nApIaf[muscle][0])/self._nIaf/self._timeWindowRec*1000
            firingRates[muscle]["IIf"] = (self._nApIIf[muscle][-1]-self._nApIIf[muscle][0])/self._nIIf/self._timeWindowRec*1000
            firingRates[muscle]["Mn"] = self._muscAct[muscle]*self.maxMnFr

            activityColors[muscle] = {}
            activityColors[muscle]["IaInt"] = self._cmap(((self._nApIaInt[muscle][-1]-self._nApIaInt[muscle][0])/self._nIaInt/self._timeWindowRec*1000)/self._maxIaIntFr)
            activityColors[muscle]["IIExInt"] = self._cmap(((self._nApIIExInt[muscle][-1]-self._nApIIExInt[muscle][0])/self._nIIExInt/self._timeWindowRec*1000)/self._maxIIExIntFr)
            activityColors[muscle]["Iaf"] = self._cmap(((self._nApIaf[muscle][-1]-self._nApIaf[muscle][0])/self._nIaf/self._timeWindowRec*1000)/self._maxIafFr)
            activityColors[muscle]["IIf"] = self._cmap(((self._nApIIf[muscle][-1]-self._nApIIf[muscle][0])/self._nIIf/self._timeWindowRec*1000)/self._maxIIfFr)
            activityColors[muscle]["Mn"] = self._cmap(self._muscAct[muscle])

        #self._plotter.update_activity(activityColors,firingRates)

    def _extract_results(self):
        """ Extract the simulation results. """
        if self._debug: print "Extracting the results... ",
        actionPotentials = self._nn.extract_action_potentials()
        firings = {}
        self._meanFr = {}
        self._estimatedEMG = {}
        self._nSpikes = {}
        self._nActiveCells = {}
        for muscle in actionPotentials:
            firings[muscle]={}
            self._meanFr[muscle]={}
            self._estimatedEMG[muscle]={}
            self._nSpikes[muscle]={}
            self._nActiveCells[muscle]={}
            for cellName in actionPotentials[muscle]:
                firings[muscle][cellName] = tlsf.exctract_firings(actionPotentials[muscle][cellName].spiketrains,sim.get_current_time())
                self._nActiveCells[muscle][cellName] = np.count_nonzero(np.sum(firings[muscle][cellName],axis=1))
                self._nSpikes[muscle][cellName] = np.sum(firings[muscle][cellName])
                self._meanFr[muscle][cellName] = tlsf.compute_mean_firing_rate(firings[muscle][cellName])
                if cellName in self._nn.get_motoneurons_names():
                    self._estimatedEMG[muscle][cellName] = tlsf.synth_rat_emg(firings[muscle][cellName])
        if self._debug: print "...completed."

    def _save_results(self):
        with open(self._resultsFolder+self._resultsName+".p", 'w') as pickle_file:
            pickle.dump(self._meanFr, pickle_file)
            pickle.dump(self._estimatedEMG, pickle_file)

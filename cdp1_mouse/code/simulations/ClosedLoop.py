import pyNN.nest as sim
from Simulation import Simulation
from AfferentFibersPopulation import AfferentFibersPopulation
from multiprocessing import Process, Pipe
import random as rnd
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from tools import firings_tools as tlsf
from tools import afferents_tools as tlsa
from tools import networkPlotter as nnplt
#from tools import RealTimePlotter as rtp


class ClosedLoop(Simulation):
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

	def __init__(self, neuralNetwork, dtCommunication, species="human", ees=None, tStop=999999, figName="networkActivity"):
		""" Object initialization.

		Keyword arguments:
		neuralNetwork -- NeuralNetwork object.
		dtCommunication -- period of time in ms at wich the firing rate is updated from the musculoskeletal model.
		species -- 'mouse', 'rat' or 'human' (default == human))
		ees -- possible EES object connected to the NeuralNetwork. If no EES objects
		are connected use None (default = None).
		tStop -- Time in ms at wich the simulation will stop (default = 999999).
		"""

		Simulation.__init__(self)
		self._debug = False
		self._plotFlag = True

		self._nn = neuralNetwork
		self._ees = ees #! Not really usefull so far...
		self._species = species
		self._figName = figName
		self._dtCommunication = dtCommunication
		self._set_tstop(tStop)

		self._init_afferents_fr()
		self._init_muscles_param()
		self._incoming_msg = ""
		self._doStep = False


		self._tMuscModel = 0
		self.maxMnFr = 50. #Hz

		if self._plotFlag: self._initialize_plot_data()

		if not self._nn.use_simple_afferent_models():
			self._set_integration_step(AfferentFibersPopulation.get_update_period())
		else: self._set_integration_step(self._dtCommunication)



	"""
	Redefinition of inherited methods
	"""
	def _initialize(self):
		Simulation._initialize(self)
		self._timeUpdateAfferents = 0

	def _integrate(self):
		""" Integrate the neuronal cells for a defined integration time step ."""
		if self._doStep: Simulation._integrate(self)

	def _update(self):
		""" Update simulation parameters. """
		time = sim.get_current_time()
		if self._doStep:
			if not self._nn.use_simple_afferent_models(): self._nn.update_afferents_ap()
			if time-self._tLastStep>= (self._dtCommunication-0.5*self._get_integration_step()):
				if self._debug: print "dt integration completed."
				if self._debug: print "compute musc activations..."
				self._compute_musc_act()
				if self._debug: print "send data.."
				self._send_data()
				self._doStep = False
				if self._plotFlag: self.plot()
		else:
			if self._debug: print "Waiting for a message..." # The Comm in is not compleatly necessary
			self._incoming_msg = sys.stdin.readline().rstrip("\n")
			if self._debug: print "Message received: "+str(self._incoming_msg)
			if self._incoming_msg == "COMM IN": self._doStep = True
			if self._doStep:
				if self._debug: print "doing a step of simulation..."
				self._tLastStep = time
				self._get_data()
				if self._debug: print "data received..."
				self._update_afferents_fr()
				self._nn.set_afferents_fr(self._afferentFr)
				if self._debug: print "afferents updated..."

	def _end_integration(self):
		""" Print the total simulation time and extract the results. """
		Simulation._end_integration(self)
		self._extract_results()
		if self._plotFlag: self._plotter.end_plotting()

	def plot(self):

		nApIaIntNew = self._nn.get_ap_number(["IaInt"])
		nApIIExIntNew = self._nn.get_ap_number(["IIExInt"])
		nApIafNew = self._nn.get_ap_number(["Iaf"])
		nApIIfNew = self._nn.get_ap_number(["IIf"])


		IaIntFr={}
		IIExIntFr={}
		IIfFr={}
		IafFr={}

		activityColors = {}
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

			print ((self._nApIaf[muscle][-1]-self._nApIaf[muscle][0])/self._nIaf/self._timeWindowRec*1000)

			activityColors[muscle] = {}
			activityColors[muscle]["IaInt"] = self._cmap(((self._nApIaInt[muscle][-1]-self._nApIaInt[muscle][0])/self._nIaInt/self._timeWindowRec*1000)/self._maxIaIntFr)
			activityColors[muscle]["IIExInt"] = self._cmap(((self._nApIIExInt[muscle][-1]-self._nApIIExInt[muscle][0])/self._nIIExInt/self._timeWindowRec*1000)/self._maxIIExIntFr)
			activityColors[muscle]["Iaf"] = self._cmap(((self._nApIaf[muscle][-1]-self._nApIaf[muscle][0])/self._nIaf/self._timeWindowRec*1000)/self._maxIafFr)
			activityColors[muscle]["IIf"] = self._cmap(((self._nApIIf[muscle][-1]-self._nApIIf[muscle][0])/self._nIIf/self._timeWindowRec*1000)/self._maxIIfFr)
			activityColors[muscle]["Mn"] = self._cmap(self._muscAct[muscle])

		self._plotter.update_activity(activityColors)


	"""
	Specific Methods of this class
	"""

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

	def _get_data(self):
		""" Read and update muscles stretch in mm and time in ms form the musculoskeletal model. """

		# update values of the previous step
		for muscle in self._muscStretch: self._muscStretchOld[muscle]=self._muscStretch[muscle]
		tMuscModelOld = self._tMuscModel

		# Read the current time in the musculoskeletal model.
		self._tMuscModel = float(sys.stdin.readline().rstrip("\n"))
		dtMuscModel = self._tMuscModel-tMuscModelOld # this dt should be equal to the closedLoop period of the neural network
		# assert abs(dtMuscModel-self._dtCommunication)<1.5, "mismatch between the NN dt and the MM dt: %f vs %f"%(self._dtCommunication,dtMuscModel)

		# Read and update the muscles stretch from the musculoskeletal model.
		for i in self._muscStretch:
			# The string coming from the musculoskeletal model has to be formatted as: "muscle(str) stretch(float) in m"
			muscInfo = sys.stdin.readline().rstrip("\n").split()
			muscle = muscInfo[0]
			self._muscStretch[muscle] = float(muscInfo[1])*1000 # transform it in mm

		# Update the muscles stretch velocity
		if sim.get_current_time() != 0:
			for muscle in self._muscStretch:
				self._muscStretchVel[muscle] = (self._muscStretch[muscle]-self._muscStretchOld[muscle])/dtMuscModel

	def _update_afferents_fr(self):
		""" Estimate and update the afferents firing rate."""
		for muscle in self._afferentFr:
			# compute the afferents fr
			self._afferentFr[muscle]['Iaf'] = tlsa.compute_Ia_fr(self._muscStretch[muscle],self._muscStretchVel[muscle],self._muscAct[muscle],self._species)
			self._afferentFr[muscle]['IIf'] = tlsa.compute_II_fr(self._muscStretch[muscle],self._muscAct[muscle],self._species)


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

	def _send_data(self):
		print "COMM_OUT"
		for muscle in self._muscAct: print " ".join([muscle,str(self._muscAct[muscle])])
		print "END"

	def _initialize_plot_data(self):
		self._cmap = plt.get_cmap('viridis')
		# self._cmap = plt.get_cmap('rainbow')
		self._maxIaIntFr = 150.
		self._maxIIExIntFr = 150.
		self._maxIafFr = 150.
		self._maxIIfFr = 150.
		movie = {'folder':self._resultsFolder,
				'name':self._figName,
				'fps':int(1000./self._dtCommunication)}
		self._plotter = nnplt.musclespindles_network_plotter("LEFT_LG","LEFT_TA",None)

	def _extract_results(self):
		""" Extract the simulation results. """
		print "Extracting the results... ",
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
				firings[muscle][cellName] = tlsf.exctract_firings(actionPotentials[muscle][cellName].spiketrains,self._get_tstop())
				self._nActiveCells[muscle][cellName] = np.count_nonzero(np.sum(firings[muscle][cellName],axis=1))
				self._nSpikes[muscle][cellName] = np.sum(firings[muscle][cellName])
				self._meanFr[muscle][cellName] = tlsf.compute_mean_firing_rate(firings[muscle][cellName])
				if cellName in self._nn.get_motoneurons_names():
					self._estimatedEMG[muscle][cellName] = tlsf.synth_rat_emg(firings[muscle][cellName])
		print "...completed."

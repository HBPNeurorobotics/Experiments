import pyNN.nest as sim
from CellsParameters import CellsParameters
from AfferentFibersPopulation import AfferentFibersPopulation
from AfferentFibersPopulationSimple import AfferentFibersPopulationSimple
import time
import numpy as np

import os
pathToCode = os.path.dirname(__file__)

class SpinalNeuralNetwork():
	""" Spiking neural network model.

	Model of a spiking neural network.
	"""

	def __init__(self, inputFile):
		""" Object initialization.

		Keyword arguments:
		inputFile -- txt file specifying the neural network structure.
		"""

		self._inputFile = inputFile

		# Flag to decide wether to use the simple afferent model or not
		self._simpleAfferentModel = True

		# Percentage of noise on synaptic weights and delay
		self._noisePerc = 0.1

		# Initialize the flags to decide which neurons to record.
		self.recordMotoneurons = True
		self.recordAfferents = True
		self.recordInterneurons = True

		# Initialize the lists containing the names of the different types of cells.
		self._motoneuronsNames = []
		self._primaryAfferentsNames = []
		self._secondaryAfferentsNames = []
		self._afferentsNames = []
		self._interneuronsNames = []

		self._projections = []

		# Build the neural network
		self._read()
		self._init_dictionaries()
		self._create_cells()
		self._create_common_connections()
		self._create_inter_muscles_sensorimotor_connections()
		self._create_special_connections()

	def __del__(self):
		""" Object destruction. """
		pass

	def _read(self):
		""" Define the neural network structure from the input file. """
		self._infoMuscles = []
		self._infoCommonCellsInMuscles = []
		self._infoSpecialCells = []
		self._infoCommonMuscleConnections = []
		self._infoInterMuscSensorimotorConnections = {}
		self._infoSpecialConnections = []

		section = None
		sensorimotorConnections = None
		sensorimotorMatrix = None
		for line in open(pathToCode+"/nnStructures/"+self._inputFile,"r"):
			if line[0] == "#" or line[0] == "\n": continue
			elif line[0] == "@": section = float(line[1])
			elif section == 1: self._infoMuscles.append(line.strip("\n").split())
			elif section == 2: self._infoCommonCellsInMuscles.append(line.strip("\n").split())
			elif section == 3: self._infoSpecialCells.append(line.strip("\n").split())
			elif section == 4: self._infoCommonMuscleConnections.append(line.strip("\n").split())
			elif section == 5:
				if line[0] == "+":
					dictName = line[1:].strip("\n")
					self._infoInterMuscSensorimotorConnections[dictName] = {}
					sensorimotorConnections = False
					sensorimotorMatrix = False
				elif "Connections" in line:
					 sensorimotorConnections = True
					 self._infoInterMuscSensorimotorConnections[dictName]["connections"]=[]
				elif "WeightsMatrix" in line:
					 sensorimotorConnections = False
					 sensorimotorMatrix = True
					 self._infoInterMuscSensorimotorConnections[dictName]["matrix"]=[]
				elif sensorimotorConnections:
					self._infoInterMuscSensorimotorConnections[dictName]["connections"].append(line.strip("\n").split())
				elif sensorimotorMatrix:
					self._infoInterMuscSensorimotorConnections[dictName]["matrix"].append(line.strip("\n").split())
			elif section == 6: self._infoSpecialConnections.append(line.strip("\n").split())

	def _init_dictionaries(self):
		""" Initialize all the dictionaries contatining cells, cell ids and the recorded action potentials. """
		# Dictionary containing all cells and cell labels
		self.cells = {}
		self.cellLabels = {}
		self._recorder_device = {}
		self._nMuscles = len(self._infoMuscles)
		for muscle,muscAfferentDelay in self._infoMuscles:
			# Create list/assembly for all muscles
			self.cellLabels[muscle]=[]
			self.cells[muscle]=sim.Assembly(label=muscle)
			self._recorder_device[muscle] = {}
		# Add special cells (specifc for some muscles or not muscle related)
		for cellInfo in self._infoSpecialCells:
			groupOrMuscle = cellInfo[0]
			if not groupOrMuscle in self.cellLabels.keys():
				self.cellLabels[groupOrMuscle]=[]
				self.cells[groupOrMuscle]=sim.Assembly(label=groupOrMuscle)
				self._recorder_device[groupOrMuscle] = {}

	def _create_cells(self):
		""" Create the desired cell populations. """
		# Iterate over all dictionaries
		for muscle,muscAfferentDelay in self._infoMuscles:
			for cellInfo in self._infoCommonCellsInMuscles:
				cellClass = cellInfo[0]
				cellTypeName = cellInfo[1]
				cellName = cellInfo[2]
				cellNumber = int(cellInfo[3])
				if len(cellInfo)>=5: neuronParam = cellInfo[4]
				else: neuronParam = None
				self.cellLabels[muscle].append(cellName)
				self._create_cell_population(muscle,muscAfferentDelay,cellClass,cellTypeName,cellName,cellNumber,neuronParam)
		# Add special cells
		for cellInfo in self._infoSpecialCells:
			groupOrMuscle = cellInfo[0]
			cellClass = cellInfo[1]
			cellTypeName = cellInfo[2]
			cellName = cellInfo[3]
			cellNumber = int(cellInfo[4])
			if len(cellInfo)>=6: neuronParam = cellInfo[5]
			else: neuronParam = None
			muscAfferentDelay = None
			self.cellLabels[groupOrMuscle].append(cellName)
			self.cells[groupOrMuscle]=sim.Assembly(label=groupOrMuscle)
			self._create_cell_population(groupOrMuscle,muscAfferentDelay,cellClass,cellTypeName,cellName,cellNumber,neuronParam)
		self._afferentsNames = self._primaryAfferentsNames+self._secondaryAfferentsNames

	def _create_cell_population(self,muscle,muscAfferentDelay,cellClass,cellTypeName,cellName,cellNumber,neuronParam=None):
		""" Create cells populations. """
		if cellClass=="motoneuron":
			#List containing all integrate and fire motoneurons names
			if not cellName in self._motoneuronsNames: self._motoneuronsNames.append(cellName)
			cellPopulation = sim.Population(cellNumber,CellsParameters.motoneurons(cellTypeName),label=cellName)
		elif cellClass=="afferentFiber":
			#Lists containing all primary or secondary afferent fibers names
			if "II" in cellName:
				if not cellName in self._secondaryAfferentsNames: self._secondaryAfferentsNames.append(cellName)
			else:
				if not cellName in self._primaryAfferentsNames: self._primaryAfferentsNames.append(cellName)
			# delay - parameter specific for the Afferent fibers
			if neuronParam is not None: delay = int(neuronParam)
			elif muscAfferentDelay is not None: delay = int(muscAfferentDelay)
			else: raise Exception("Please specify the afferent fiber delay")

			if self._simpleAfferentModel:
				cellPopulation = AfferentFibersPopulationSimple(cellNumber,delay,label=cellName)
			else:
				cellPopulation = AfferentFibersPopulation(cellNumber,delay,label=cellName)
		elif cellClass=="interneuron":
			#List containing all interneurons names
			if not cellName in self._interneuronsNames: self._interneuronsNames.append(cellName)
			cellPopulation = sim.Population(cellNumber,CellsParameters.interneurons(cellTypeName),label=cellName)
		else:
			raise Exception("Unkown cellClass in the netowrk instructions.... ("+str(cellClass)+")")
		# Record cells APs
		if (cellClass=="motoneuron" and self.recordMotoneurons)\
			or (cellClass=="afferentFiber" and self.recordAfferents)\
			or (cellClass=="interneuron" and self.recordInterneurons):
			# To speed up the simulation we don't save the spikes to a file
			cellPopulation.record('spikes',to_file=False)
			self._recorder_device[muscle][cellName] = cellPopulation.recorder._spike_detector.device
			sim.nest.SetStatus(self._recorder_device[muscle][cellName], "to_memory", True)
			sim.nest.SetStatus(self._recorder_device[muscle][cellName], "to_file", False)

		self.cells[muscle] += cellPopulation

	def _create_common_connections(self):
		""" Connect network cells within the same degree of freedom. """
		for muscle,muscAfferentDelay in self._infoMuscles:
			for connection in self._infoCommonMuscleConnections:
				# Presynaptic cells
				presynapticCells = self.cells[muscle].get_population(connection[0])
				# Postsynaptic cells
				postsynapticCells = self.cells[muscle].get_population(connection[1])
				# Connection probability (0-1]
				conProbability = float(connection[2])
				# Weight of connections
				conWeight = float(connection[3])*7
				# Type of synapse
				receptorType = connection[4]
				# connect sources to targets
				w = sim.random.RandomDistribution('normal', [conWeight, np.abs(conWeight)*self._noisePerc], rng=sim.random.NumpyRNG(seed=4242))
				delay = sim.random.RandomDistribution('normal_clipped', [1, 1*self._noisePerc,0.1,2], rng=sim.random.NumpyRNG(seed=4241))
				syn = sim.StaticSynapse(weight=w,delay=delay)
				if conProbability >= 1: connector = sim.AllToAllConnector()
				else: connector = sim.FixedProbabilityConnector(p_connect=conProbability)
				proj = sim.Projection(presynapticCells, postsynapticCells, connector, syn, receptor_type=receptorType)
				self._projections.append(proj)


	def _create_inter_muscles_sensorimotor_connections(self):
		""" Create sensorimotor connections between muscles."""
		for pathway in self._infoInterMuscSensorimotorConnections:
			connections = self._infoInterMuscSensorimotorConnections[pathway]["connections"]
			matrix = self._infoInterMuscSensorimotorConnections[pathway]["matrix"]
			if not len(matrix)-1 == len(matrix[0])-1 == len(self._infoMuscles):
				raise(Exception("The weight matrix has to be nMuscles x nMuscles."))
			# The first raw is a header
			for M2weights,M1 in zip(matrix[1:],self._infoMuscles):
				for weight,M2 in zip(M2weights[1:],self._infoMuscles):
					if not float(weight) == 0:
						if M1[0] is M2[0]: raise(Exception("Intra muscle sensorimotor conncetions have to be implemented in section 4."))
						for connection in connections:
							# Presynaptic cells
							presynapticCells = self.cells[M1[0]].get_population(connection[1])
							# Postsynaptic cells
							postsynapticCells = self.cells[M2[0]].get_population(connection[3])
							# Connection probability (0-1]
							conProbability = float(connection[4])*float(weight)
							# Weight of connections
							conWeight = float(connection[5])*7
							# Type of synapse
							receptorType = connection[6]
							# connect sources to targets
							w = sim.random.RandomDistribution('normal', [conWeight, np.abs(conWeight)*self._noisePerc], rng=sim.random.NumpyRNG(seed=4242))
							delay = sim.random.RandomDistribution('normal_clipped', [1, 1*self._noisePerc,0.1,2], rng=sim.random.NumpyRNG(seed=4241))
							syn = sim.StaticSynapse(weight=w,delay=delay)
							if conProbability >= 1: connector = sim.AllToAllConnector()
							else: connector = sim.FixedProbabilityConnector(p_connect=conProbability)
							proj = sim.Projection(presynapticCells, postsynapticCells, connector, syn, receptor_type=receptorType)
							self._projections.append(proj)


	def _create_special_connections(self):
		""" Create connections specific to single muscles or cell groups. """
		for connection in self._infoSpecialConnections:
			# Presynaptic cells
			presynapticCells = self.cells[connection[0]].get_population(connection[1])
			# Postsynaptic cells
			postsynapticCells = self.cells[connection[2]].get_population(connection[3])
			# Connection probability (0-1]
			conProbability = float(connection[4])
			# Weight of connections
			conWeight = float(connection[5])*7
			# Type of synapse
			receptorType = connection[6]
			# connect sources to targets
			w = sim.random.RandomDistribution('normal', [conWeight, np.abs(conWeight)*self._noisePerc], rng=sim.random.NumpyRNG(seed=4242))
			delay = sim.random.RandomDistribution('normal_clipped', [1, 1*self._noisePerc,0.1,2], rng=sim.random.NumpyRNG(seed=4241))
			syn = sim.StaticSynapse(weight=w,delay=delay)
			if conProbability >= 1: connector = sim.AllToAllConnector()
			else: connector = sim.FixedProbabilityConnector(p_connect=conProbability)
			proj = sim.Projection(presynapticCells, postsynapticCells, connector, syn, receptor_type=receptorType)
			self._projections.append(proj)

	def update_afferents_ap(self):
		""" Update all afferent fibers ation potentials. """
		if not self._simpleAfferentModel:
			for muscle in self.cells:
				for cellName in self._afferentsNames:
					self.cells[muscle].get_population(cellName).update()

	def set_afferents_fr(self,fr):
		""" Set the firing rate of the afferent fibers.

		Keyword arguments:
		fr -- Dictionary with the firing rate in Hz for the different cellNames.
		"""
		for muscle in self.cells:
			for cellName in self.cellLabels[muscle]:
				if cellName in self._afferentsNames:
					self.cells[muscle].get_population(cellName).set_firing_rate(fr[muscle][cellName])

	def initialise_afferents(self):
		""" Initialise cells parameters. """
		for muscle in self.cells:
			for cellName in self.cellLabels[muscle]:
				if cellName in self._afferentsNames:
					self.cells[muscle].get_population(cellName).initialize_cells_activity()

	def extract_action_potentials(self):
		if 'actionPotentials' in dir(self): return self.actionPotentials
		self.actionPotentials={}
		for muscle in self.cells:
			self.actionPotentials[muscle]={}
			for cellName in self.cellLabels[muscle]:
				if (cellName in self._afferentsNames and self.recordAfferents)\
					or (cellName in self._motoneuronsNames and self.recordMotoneurons)\
					or (cellName in self._interneuronsNames and self.recordInterneurons):
					data = self.cells[muscle].get_population(cellName).get_data()
					self.actionPotentials[muscle][cellName] = data.segments[0]
		return self.actionPotentials

	def get_ap_number(self, cellNames):
		""" Return the number of action potentials fired for the different recorded cells.

		The number of Ap is returned only to the main process (rank=0).
		Keyword arguments:
		cellNames -- List of cell names from wich we want to get the number of action potentials. """
		# I should modfy it to take as inputs network and muscle.

		apNumber = {}
		for muscle in self.cells:
			apNumber[muscle] = {}
			for cellName in cellNames:
				if (cellName in self._afferentsNames and self.recordAfferents) \
				or (cellName in self._motoneuronsNames and self.recordMotoneurons) \
				or (cellName in self._interneuronsNames and self.recordInterneurons):
					# data = self.cells[muscle].get_population(cellName).get_data()
					# apNumber[muscle][cellName] = sum([len(spikeTrain) for spikeTrain in data.segments[0].spiketrains])

					nest_info = sim.nest.GetStatus(self._recorder_device[muscle][cellName], 'events')[0]
					apNumber[muscle][cellName] = nest_info['times'].size
				else: raise(Exception("Cell name not found in the NeuralNetwork"))
		return apNumber

	def use_simple_afferent_models(self):
		return self._simpleAfferentModel

	def get_afferents_names(self):
		""" Return the afferents name. """
		return self._afferentsNames

	def get_primary_afferents_names(self):
		""" Return the primary afferents name. """
		return self._primaryAfferentsNames

	def get_secondary_afferents_names(self):
		""" Return the secondary afferents name. """
		return self._secondaryAfferentsNames

	def get_motoneurons_names(self):
		""" Return the motoneurons names. """
		return self._motoneuronsNames

	def get_interneurons_names(self):
		""" Return the inteurons names. """
		return self._interneuronsNames

	def get_mn_info(self):
		""" Return the connection informations. """
		return self._infoCommonMuscleConnections, self._infoSpecialConnections

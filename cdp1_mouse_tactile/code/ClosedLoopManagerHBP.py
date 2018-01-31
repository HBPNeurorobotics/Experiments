import subprocess
import socket

import os
pathToCode = os.environ['NN_CODE']

class ClosedLoopManager():
	""" Simulation Manager.

	Run the neural network processes and manage the inter process communication.
	"""

	def __init__(self, nnNp, eesFreq, eesAmp,nnStructFile,species,totSimulationTime,figName):
		# Define neural network parameters
		self._debug = False
		self._np = nnNp
		self._eesFreq = eesFreq
		self._eesAmp = eesAmp
		self._totSimulationTime = totSimulationTime
		self._nnStructFile = nnStructFile
		self._species = species
		self._figName = figName
		self._run_neural_network()

	def _run_neural_network(self):
		""" Run the neural network as a subprocess. """

		program = ['python',pathToCode+'/scripts/runClosedLoopNn.py',self._eesFreq,self._eesAmp,self._nnStructFile,self._species,self._totSimulationTime,self._figName ]

		if self._debug: print "Running: "+" ".join(program)
		self._neuralNetwork = subprocess.Popen(program, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	def run_step(self,mmData):
		if self._debug: print "sending data to nn:"
		self._send_data_to_nn(mmData)
		if self._debug: print "reading data from nn:"
		nnData = self._nn_read_data()
		if self._neuralNetwork.poll() is None: return nnData
		else: return None

	def _send_data_to_nn(self,wbtData):
		""" Send webots' data to the neural network. """
                self._neuralNetwork.stdin.flush()
		self._neuralNetwork.stdin.write("COMM IN\n") # this shitty COMM IN is not really needed..to modify in closedloop.py
                self._neuralNetwork.stdin.flush()
                self._neuralNetwork.stdin.write(wbtData)
                self._neuralNetwork.stdin.flush()


	def _nn_read_data(self):
		""" Read the data coming form the neural network. """
		reaData = True
		nnIncomingData = False
		nnData = ""
		while reaData and self._neuralNetwork.poll()==None:
			nnIncomingMsg =  self._neuralNetwork.stdout.readline().rstrip("\n").split()
			if "COMM_OUT" in nnIncomingMsg: nnIncomingData = True
			elif "END" in nnIncomingMsg: reaData = False
			elif nnIncomingData: nnData += " ".join(nnIncomingMsg)+"\n"
			if self._debug: print "\t\tPyNN: "+" ".join(nnIncomingMsg)
		return nnData

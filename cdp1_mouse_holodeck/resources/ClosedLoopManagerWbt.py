import subprocess
import socket

class ClosedLoopManager():
	""" Simulation Manager.

	Run the musculoskeletal model and the neural network processes
	and manage the inter process communication (IPC).
	"""

	def __init__(self, nnNp, eesFreq, eesAmp,nnStructFile,experiment,species,totSimulationTime,figName):
		# Define neural network parameters
		self._np = nnNp
		self._eesFreq = eesFreq
		self._eesAmp = eesAmp
		self._totSimulationTime = totSimulationTime
		self._nnStructFile = nnStructFile
		self._species = species
		self._figName = figName
		self._experiment = experiment

	def _run_neural_network(self):
		""" Run the neural network as a subprocess. """
		program = ['mpiexec','-np',self._np,'python','./scripts/runClosedLoopNn.py',self._eesFreq,self._eesAmp,self._nnStructFile,self._species,self._totSimulationTime,self._figName ]
		# program = ['mpiexec','-np',self._np,'python','-m','cProfile','-s','cumtime','./scripts/runClosedLoopNn.py',self._eesFreq,self._eesAmp,self._nnStructFile,self._species,self._totSimulationTime,self._figName ]
		self._neuralNetwork = subprocess.Popen(program, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

	def _run_webots(self):
		""" Run webots as a subprocess """
		if self._experiment == "hanging":
			program = ["webots","--stdout","../../sml/worlds/852_mouse_hindlimb_SCALE_tester.wbt"]
		else: raise(Exception("Unknown experiment!"))

		self._webots = subprocess.Popen(program, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

		# Initialize TCP communication (server side)
		TCP_IP = '127.0.0.1'
		TCP_PORT = 5005
		BUFFER_SIZE = 512
		self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self._s.bind((TCP_IP, TCP_PORT))
		self._s.listen(1)
		self._conn, addr = self._s.accept()

	def run(self):
		""" Run the closed loop simualtion. """
		self._run_neural_network()
		self._run_webots()

		# We first wait for webots inputs
		webotsTurn = True
		neuralNetworkTurn = False
		while True:
			if webotsTurn:
				print "reading data from webots:"
				wbtData = self._wbt_read_data()
				print "sending data to nn..."
				self._send_data_to_nn(wbtData)
				webotsTurn = False
				neuralNetworkTurn = True

			elif neuralNetworkTurn:

				print "reading data from nn:"
				nnData = self._nn_read_data()
				if self._neuralNetwork.poll() != None: break
				print "sending data to webots..."
				self._send_data_to_wbt(nnData)
				neuralNetworkTurn = False
				webotsTurn = True

	def _wbt_read_data(self):
		""" Read the data coming from the webots controller. """
		reaData = True
		wbtIncomingData = False
		wbtData = ""
		while reaData:
			wbtIncomingMsg =  self._webots.stdout.readline().rstrip("\n").split()
			if "COMM_OUT" in wbtIncomingMsg: wbtIncomingData = True
			elif "END" in wbtIncomingMsg: reaData = False
			elif wbtIncomingData: wbtData += " ".join(wbtIncomingMsg[1:])+"\n"
			print "\t\t\t\tWebots: :"+" ".join(wbtIncomingMsg[1:])
		return wbtData


	def _send_data_to_nn(self,wbtData):
		""" Send webots' data to the neural network. """
		self._neuralNetwork.stdin.write("COMM IN\n") # this shitty COMM IN is not really needed..to modify in closedloop.py
		self._neuralNetwork.stdin.write(wbtData)


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
			print "\t\tPyNN: "+" ".join(nnIncomingMsg)
		return nnData

	def _send_data_to_wbt(self,nnData):
		""" Send the neural network's data to the webots controller. """
		nnData += "END\n"
		self._conn.send(nnData)

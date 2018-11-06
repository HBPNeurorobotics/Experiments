import sys
sys.path.append('../code')
from simulations import ClosedLoopSimulation
from tools import structures_tools as tls
import time as timeModule
import numpy as np

""" This program executes a neural-biomechanical closed loop simulation.
The program launches the neural netwrok (python) and the biomechanical
model (cpp) and manages the communication between these two programs.
The program doesn't have to be executed by MPI.
"""


eesFreq = 40
eesAmp = 235
species = "mouse"
figName  = "closedLoopHBP"
nnStructFile = "closedLoopMouse.txt"

clm = ClosedLoopSimulation(nnStructFile, species, eesAmp, eesFreq, figName)


def transfer_function(time):
	mmData = read_data_from_gazebo(time)
	nnData = clm.run_step(mmData)
	# if nnData is not None: send_data_to_gazebo(nnData)
	# else: terminate_experiment()
	send_data_to_gazebo(nnData)

def read_data_from_gazebo(time):
	# To implement with ROS
	mmData = {'t':time, 'stretch':{}}
	mmData['stretch']['LEFT_LG'] = 10+5*np.random.rand(1)
	mmData['stretch']['LEFT_TA'] = 0
	return mmData

def send_data_to_gazebo(nnData):
	# To implement with ROS
	pass

def terminate_experiment():
	pass


if __name__ == '__main__':
	time=0
	totTime =10000
	dt = 20
	while time<int(totTime):
		time+=dt
		timeModule.sleep(1)
		transfer_function(time)

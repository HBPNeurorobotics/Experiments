import sys
sys.path.append('../code')
import subprocess
from tools import structures_tools as tls
import numpy as np
from tools import general_tools as gt

pathToResults = "/tmp/results"

def main():
	""" This program launches a parameters systematic search for a ForSimMuscleSpindles.
	The different parameters that are changed over time are directly defined in the main function.
	The program doesn't have to be executed by MPI.
	"""
	nProc = 4
	name = "_SS_pyNN_"
	norm = '1'
	eesAmplitude = '235'
	eesFrequency = '40'
	delay = 2
	weights_1 = np.array([0.091])
	weights_2 = np.linspace(0.03,0.1,20)
	weights_3 = np.array([0.046])
	weights_4 = np.linspace(-.001,-.010,20)
	weights_5 = np.array([-0.002])
	simTime = 2000
	nSim = len(weights_1)*len(weights_2)*len(weights_3)*len(weights_4)*len(weights_5)
	count=0.
	percLastPrint=0.
	printPeriod = 0.05

	for w1 in weights_1:
		for w2 in weights_2:
			for w3 in weights_3:
				for w4 in weights_4:
					for w5 in weights_5:
						resultName = name+"_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f" % (w1,w2,w3,w4,w5)
						resultFile = gt.find("*"+resultName+"*.p",pathToResults)
						if not resultFile:
							inputFile = "generatedStructures/ss_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f.txt" % (w1,w2,w3,w4,w5)
							tls.modify_network_structure("templateFrwSim.txt",inputFile,delay,[w1,w2,w3,w4,w5])
							gt.run_subprocess(['python','./scripts/runForSimMuscleSpindles.py',eesFrequency,eesAmplitude,\
							norm,inputFile,resultName,str(simTime)])

						count+=1
						if count/nSim-percLastPrint>=printPeriod:
							percLastPrint=count/nSim
							print str(round(count/nSim*100))+"% of simulations performed..."

if __name__ == '__main__':
	main()

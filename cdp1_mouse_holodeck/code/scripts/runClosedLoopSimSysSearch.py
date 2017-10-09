import sys
import os
pathToCode = os.environ['NN_CODE']
sys.path.append(pathToCode)
import subprocess
from tools import structures_tools as tls
import numpy as np
from tools import general_tools as gt

pathToResults = pathToCode+"/../../results/"

def main():
	""" This program launches a parameters systematic search for a ClosedLoopSimWebots.
	The different parameters that are changed over time are directly defined in the main function.
	"""

	name = "SS_pyNN_closedLoop_webots"
	eesAmplitudes = ["1","240"]
	eesFrequency = "40"
	delay = "2"
	weights_1 = np.linspace(0.05,0.1,5)
	weights_2 = np.linspace(0.01,0.05,5)
	weights_3 = np.linspace(0.01,0.1,10)

	w4 = -0.00145
	w5 = -0.0045

	simTime = "3000"
	nSim = len(weights_1)*len(weights_2)*len(weights_3)*len(eesAmplitudes)
	count=0.
	percLastPrint=0.
	printPeriod = 0.05

	for w1 in weights_1:
		for w2 in weights_2:
			for w3 in weights_3:
				for eesAmplitude in eesAmplitudes:
					resultName = name+"_eesAmp_%d_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f" % (int(eesAmplitude),w1,w2,w3,w4,w5)
					resultFile = gt.find("*"+resultName+"*.p",pathToResults)
					if not resultFile:
						inputFile = "generatedStructures/ss_cl_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f.txt" % (w1,w2,w3,w4,w5)
						tls.modify_network_structure("templateClosedLoop2Dof.txt",inputFile,delay,[w1,w2,w3,w4,w5])
						program = ['python','./scripts/runClosedLoopSim.py',eesFrequency,eesAmplitude,"hanging","mouse",simTime,resultName,inputFile]
						gt.run_subprocess(program)

					count+=1
					if count/nSim-percLastPrint>=printPeriod:
						percLastPrint=count/nSim
						print str(round(count/nSim*100))+"% of simulations performed..."

if __name__ == '__main__':
	main()

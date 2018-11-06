import sys
import os
pathToCode = os.environ['NN_CODE']
pathToResults = pathToCode+"/../../results/"
sys.path.append(pathToCode)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools import general_tools as gt

def main():
	""" Open the plot of a Forward systematic search analysis. """

	name = "SS_pyNN_closedLoop_webots"
	eesAmplitudes = ["1","240"]
	eesFrequency = "40"
	delay = "2"
	weights_1 = np.linspace(0.05,0.1,5)
	weights_2 = np.linspace(0.01,0.05,5)
	weights_3 = np.linspace(0.01,0.1,10)

	w4 = -0.00145
	w5 = -0.0045

	for w3 in weights_3:
		for eesAmplitude in eesAmplitudes:
			fig, ax = plt.subplots(weights_1.size, weights_2.size,figsize=(8,4.5),sharex='col',sharey='col')
			plt.suptitle(str(eesAmplitude))
			for i1,w1 in enumerate(weights_1):
				for i2,w2 in enumerate(weights_2):
					resultName = name+"_eesAmp_%d_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f" % (int(eesAmplitude),w1,w2,w3,w4,w5)
					resultFile = gt.find("*"+resultName+"*.p",pathToResults)
					if not resultFile: continue
					with open(resultFile[0], 'r') as pickle_file:
						meanFr = pickle.load(pickle_file)

					ax[i1,i2].plot(meanFr['LEFT_TA']['Mn'],'r')
					ax[i1,i2].plot(meanFr['LEFT_LG']['Mn'],'b')
					ax[i1,i2].xaxis.set_ticklabels([])
					ax[i1,i2].yaxis.set_ticklabels([])
					ax[i1,i2].xaxis.set_ticks([])
					ax[i1,i2].yaxis.set_ticks([])
					if i2==0:ax[i1,i2].set_ylabel("w1\n%.4f"%(w1),fontsize=6)
					if i1==weights_1.size-1:ax[i1,i2].set_xlabel("w2\n%.4f"%(w2),fontsize=6)
		plt.show()


if __name__ == '__main__':
	main()

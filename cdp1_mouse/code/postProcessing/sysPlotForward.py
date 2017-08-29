import sys
sys.path.append('../code')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tools import general_tools as gt

def main():
	""" Open the plot of a Forward systematic search analysis. """

	pathToResults = "/tmp/results/"
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

	fig, ax = plt.subplots(weights_2.size, weights_4.size,figsize=(22,12),sharex='col',sharey='col')
	for w1 in weights_1:
		for i2,w2 in enumerate(weights_2):
			for w3 in weights_3:
				for i4,w4 in enumerate(weights_4):
					for w5 in weights_5:
						pattern = "*%suA_%sHz_Delay_%dms%s_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f.p" % (eesAmplitude,eesFrequency,delay,name,w1,w2,w3,w4,w5)
						resultFile = gt.find(pattern,pathToResults)
						if not resultFile: continue
						with open(resultFile[0], 'r') as pickle_file:
							_ = pickle.load(pickle_file)
							meanFr = pickle.load(pickle_file)

						ax[i2,i4].plot(meanFr['TA']['IaInt'],'r')
						ax[i2,i4].plot(meanFr['GM']['IaInt'],'b')
						ax[i2,i4].xaxis.set_ticklabels([])
						ax[i2,i4].yaxis.set_ticklabels([])
						ax[i2,i4].xaxis.set_ticks([])
						ax[i2,i4].yaxis.set_ticks([])
						if i4==0:ax[i2,i4].set_ylabel("w2\n%.4f"%(w2),fontsize=6)
						if i2==weights_2.size-1:ax[i2,i4].set_xlabel("w4\n%.4f"%(w4),fontsize=6)
	plt.show()

if __name__ == '__main__':
	main()

import matplotlib.pyplot as plt
import time
import numpy as np

class RealTimePlotter():
	""" Plot in real time. """

	def __init__(self, yLabels, ylimMaxVals, tStop=3000):

		size = 0.5
		self.yLabels = yLabels
		self._fig, self._ax = plt.subplots(len(self.yLabels),1,figsize=(16*size,9*size))
		self._resultsFolder = "/tmp/results/"
		if type(self._ax) is not np.ndarray: self._ax = np.array([self._ax])
		for ax, yLabel,ylimMax in zip(self._ax,self.yLabels,ylimMaxVals):
			ax.set_xlim([0, tStop])
			ax.set_ylim([0, ylimMax])
			ax.set_ylabel(yLabel,fontsize=6)
			# Move left spine outward by 5 points
			ax.spines['left'].set_position(('outward', 5))
			# Hide the right, bottom and top spines
			ax.spines['right'].set_visible(False)
			ax.spines['top'].set_visible(False)
			# Only show ticks on the left spine
			ax.yaxis.set_ticks_position('left')
			ax.xaxis.set_ticks_position('none')
			ax.tick_params(axis='both', which='major', labelsize=6,labelbottom='off')

		ax.tick_params(axis='both', which='major', labelsize=6,labelbottom='on')
		ax.spines['bottom'].set_position(('outward', 5))
		ax.xaxis.set_ticks_position('bottom')
		ax.set_xlabel("Time (ms)")

		plt.ion()
		plt.show()

	def add_values(self,xVal,yVals):
		colors = ['#3176a8','#3ac687']
		for ax, yL in zip(self._ax, self.yLabels):
			for i,yKey in enumerate(yVals[yL]):
				ax.scatter(xVal,yVals[yL][yKey],c=colors[i])
			ax.legend(yVals[yL].keys())
		plt.draw()
		plt.pause(0.1)

	def save_fig(self,nameSuffi=""):
		fileName = time.strftime("%Y_%m_%d_RealTimePlot"+nameSuffi+".pdf")
		plt.savefig(self._resultsFolder+fileName, format="pdf",transparent=True)

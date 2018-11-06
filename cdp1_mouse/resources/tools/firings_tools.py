import numpy as np
import random as rnd
import time


def exctract_firings(spikeTrains,maxSimTime = 0, samplingRate = 1000.):
	""" Extract cells firings from different hosts and return them to host 0.

	Keyword arguments:
		apListVector -- List of action potentials Neuron vectors.
		maxTime -- Maximun time of the simulation in ms (default = 0).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
	"""

	# extctraing the action potentials time
	nAp = [spiketrain.size for spiketrain in spikeTrains]
	nCells = len(spikeTrains)

	if not nAp: maxNap = 0
	else: maxNap = max(nAp)

	# exctracting the matrix with the ap time (nCells x maxNap)
	maxTime = maxSimTime
	if nAp:
		actionPots = -1*np.ones([nCells,maxNap])
		for i,aps in enumerate(spikeTrains):
			actionPots[i,:aps.size] = aps

	# exctracting the firings matrix of 0 and 1
	dt = 1000./samplingRate
	firings = np.zeros([nCells,1+int(maxTime/dt)])
	if nAp:
		actionPots = (actionPots/dt).astype(int)
		for i in xrange(nCells):
			indx = actionPots[i,:]>=0
			firings[i,actionPots[i,indx]]=1
	return firings

# use reshape instead of concatenate...(concatenate could be usefull if not same size but not in this case)

def compute_mean_firing_rate(firings,samplingRate = 1000.):
	""" Return the mean firing rates given the cell firings.

	Keyword arguments:
		firings -- Cell firings, a 2d numpy array (nCells x time).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
	"""

	interval = 100*samplingRate/1000 #ms
	nCells = firings.shape[0]
	nSamples = firings.shape[1]

	meanFrTemp = np.zeros(nSamples)
	meanFr = np.zeros(nSamples)
	for i in xrange(int(interval),nSamples):
		totAp = firings[:,i-int(interval):i].sum()
		meanFrTemp[i-int(round(interval/2))]=totAp/nCells*samplingRate/interval

	# Smooth the data with a moving average
	windowSize = int(50*samplingRate/1000) #ms
	for i in xrange(windowSize,nSamples):
		meanFr[i-int(round(windowSize/2))] = meanFrTemp[i-windowSize:i].mean()

	return meanFr

def synth_rat_emg( firings,samplingRate = 1000.,delay_ms=2):
	""" Return the EMG activity given the cell firings.

	Keyword arguments:
		firings -- Cell firings, a 2d numpy array (nCells x time).
		samplingRate -- Sampling rate of the extracted signal in Hz (default = 1000).
		delay_ms -- delay in ms between an action potential (AP) and a motor unit
		action potential (MUAP).
	"""
	rnd.seed(time.time())
	nCells = firings.shape[0]
	nSamples = firings.shape[1]

	dt = 1000./samplingRate
	delay = int(delay_ms/dt)

	# MUAP duration between 5-10ms (Day et al 2001) -> 7.5 +-1
	meanLenMUAP = int(7.5/dt)
	stdLenMUAP = int(1/dt)
	nS = [int(meanLenMUAP+rnd.gauss(0,stdLenMUAP)) for i in xrange(firings.shape[0])]
	Amp = [abs(1+rnd.gauss(0,0.2)) for i in xrange(firings.shape[0])]
	EMG = np.zeros(nSamples + max(nS)+delay);
	# create MUAP shape
	for i in xrange(nCells):
		n40perc = int(nS[i]*0.4)
		n60perc = nS[i]-n40perc
		amplitudeMod = (1-(np.linspace(0,1,nS[i])**2)) * np.concatenate((np.ones(n40perc),1/np.linspace(1,3,n60perc)))
		logBase = 1.05
		freqMod = np.log(np.linspace(1,logBase**(4*np.pi),nS[i]))/np.log(logBase)
		EMG_unit = Amp[i]*amplitudeMod*np.sin(freqMod);
		for j in xrange(nSamples):
			if firings[i,j]==1:
				EMG[j+delay:j+delay+nS[i]]=EMG[j+delay:j+delay+nS[i]]+EMG_unit
	EMG = EMG[:nSamples]

	return EMG

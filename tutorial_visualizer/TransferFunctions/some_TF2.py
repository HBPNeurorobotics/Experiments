import hbp_nrp_cle.tf_framework as nrp
@nrp.NeuronMonitor(nrp.brain.someNeuron, nrp.spike_recorder)
def some_TF2(t):
	return True
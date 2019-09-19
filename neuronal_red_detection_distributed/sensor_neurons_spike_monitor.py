@nrp.NeuronMonitor(nrp.brain.circuit[slice(600, 608)], nrp.spike_recorder)
def sensor_neurons_spike_monitor(t):
    return True


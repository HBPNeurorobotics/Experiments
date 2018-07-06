    # Imported Python Transfer Function
    #
    @nrp.NeuronMonitor(nrp.brain.plot, nrp.spike_recorder)
    def all_neurons_spike_monitor(t):
        return True
    #

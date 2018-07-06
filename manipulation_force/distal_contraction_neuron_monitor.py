    # Imported Python Transfer Function
    #
    @nrp.NeuronMonitor(nrp.brain.monitor_neurons, nrp.spike_recorder)
    def distal_contraction_neuron_monitor(t):
        return True
    #

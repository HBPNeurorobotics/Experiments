    # Imported Python Transfer Function
    #
    # NOTE: record has 1280 neurons, monitoring slows down the sim
    # it could be useful to reduce its size e.g. [960,1280]
    @nrp.NeuronMonitor(nrp.brain.record, nrp.spike_recorder)
    def all_neurons_monitor(t):
        return True
    #

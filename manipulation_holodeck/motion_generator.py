    # Imported Python Transfer Function
    #
    @nrp.MapSpikeSource("motion_generators", nrp.brain.motor, nrp.fixed_frequency)
    @nrp.Robot2Neuron()
    def motion_generator(t, motion_generators):
        motion_generators.rate = abs(np.sin(t)) * 30
    #

    # Imported Python Transfer Function
    #
    # Compute Target position following a triangular trajectory
    @nrp.MapVariable("target_freq",scope=nrp.GLOBAL)
    @nrp.MapVariable("target_ampl", scope=nrp.GLOBAL)
    @nrp.MapVariable("target_delta", scope=nrp.GLOBAL)
    @nrp.MapVariable("trajectory", scope=nrp.GLOBAL)
    @nrp.Robot2Neuron()
    def compute_target_position_triang(t, target_freq, target_ampl, target_delta, trajectory):
        if trajectory != "triangular":
            return
        frequency = target_freq.value
        amplitude = target_ampl.value
        T = 1./frequency
        if t % T <= T/4.:
            target_delta.value = ((t % (T / 4.)) / (T / 4.)) * (float(amplitude) / 2.)
        elif t % T > T/4. and t % T <= 3*T/4.:
            target_delta.value = (float(amplitude) / 2.) - (((t - T/4.) % (T / 2.)) / (T / 2.)) * (float(amplitude))
        else:
            target_delta.value = ((t % (T / 4.)) / (T / 4.)) * (float(amplitude) / 2.) - (float(amplitude) / 2.)
    #

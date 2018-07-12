# Imported Python Transfer Function
#
# Compute Target position following a sinusoidal trajectory
@nrp.MapVariable("target_freq",scope=nrp.GLOBAL)
@nrp.MapVariable("target_ampl", scope=nrp.GLOBAL)
@nrp.MapVariable("target_delta", scope=nrp.GLOBAL)
@nrp.MapVariable("trajectory", scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def compute_target_position_sine(t, target_freq, target_ampl, target_delta, trajectory):
    if trajectory != "sinusoidal":
        return
    frequency = target_freq.value
    amplitude = target_ampl.value
    target_delta.value = np.sin(t * frequency * 2 * np.pi) * (float(amplitude) / 2)
#


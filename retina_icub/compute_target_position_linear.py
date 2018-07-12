# Imported Python Transfer Function
#
# Compute Target position following a linear trajectory
@nrp.MapVariable("target_delta", scope=nrp.GLOBAL)
@nrp.MapVariable("trajectory", scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def compute_target_position_linear(t, target_delta, trajectory):
    if trajectory != "linear":
        return
    if t <= 2 or t >= 5:
        return
    target_delta.value = 0.2*(t-2)
#


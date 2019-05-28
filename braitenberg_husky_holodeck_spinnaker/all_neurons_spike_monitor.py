@nrp.NeuronMonitor(nrp.chain_neurons(nrp.brain.red_left_eye, nrp.brain.red_right_eye, nrp.brain.green_blue_eye, nrp.brain.go_on, nrp.brain.left_wheel_motor, nrp.brain.right_wheel_motor), nrp.spike_recorder)
def all_neurons_spike_monitor(t):
    return True


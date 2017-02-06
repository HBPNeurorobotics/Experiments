@nrp.MapVariable("current_target_pan", initial_value=0.0)
@nrp.MapSpikeSink("output_neurons", nrp.map_neurons(range(0,4), lambda i: nrp.brain.motors[i]),
                  nrp.spike_recorder)
@nrp.MapRobotPublisher('head_pan', Topic('/robot/RHM0_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def move_pan(t, current_target_pan, head_pan, output_neurons):
    constant_delta_pan = 0.1
    current_delta = 0
    if output_neurons[0].spiked or output_neurons[2].spiked:
        current_delta = current_delta + constant_delta_pan
    if output_neurons[1].spiked or output_neurons[3].spiked:
        current_delta = current_delta - constant_delta_pan
    current_target_pan.value = current_target_pan.value + current_delta
    head_pan.send_message(std_msgs.msg.Float64(current_target_pan.value))

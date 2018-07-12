# Imported Python Transfer Function
#
@nrp.MapSpikeSink("output_neuron", nrp.brain.neurons[1], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/robot/hollie_real_left_arm_1_joint/cmd_pos', std_msgs.msg.Float64))
# Example TF: get output neuron voltage and actuate the arm with the current simulation time. You could do something with the voltage here and command the robot accordingly.
def turn_around(t, output_neuron):
    voltage=output_neuron.voltage
    return std_msgs.msg.Float64(t)
#


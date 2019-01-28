# Imported Python Transfer Function
#
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg
@nrp.MapSpikeSink("output_neuron", nrp.brain.neurons[1], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
# Example TF: get output neuron voltage and output constant on robot actuator. You could do something with the voltage here and command the robot accordingly.
def turn_around(t, output_neuron):
    voltage=output_neuron.voltage
    return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(0,0,0),
                                   angular=geometry_msgs.msg.Vector3(0,0,5))
#


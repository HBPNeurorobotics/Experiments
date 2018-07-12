# Imported Python Transfer Function
#
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
@nrp.MapSpikeSink("output_neuron", nrp.brain.neurons[1], nrp.leaky_integrator_alpha)
@Neuron2Robot(Topic('/voltage', std_msgs.msg.Float64))
#Example TF: Writes the voltage to a topic
def write_message_to_topic(t, output_neuron):
    # Uncomment to log into the 'log-console' visible in the simulation
    # clientLogger.info("Time: ", t)
    return output_neuron.voltage
#


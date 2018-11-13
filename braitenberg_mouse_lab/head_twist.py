# Imported Python Transfer Function
#
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
@nrp.MapSpikeSink("left_neuron", nrp.brain.actors[0], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("right_neuron", nrp.brain.actors[1], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/mouse/neck_joint/cmd_pos', std_msgs.msg.Float64))
def head_twist(t, left_neuron, right_neuron):
        voltage_right=right_neuron.voltage
        voltage_left=left_neuron.voltage
        #Calculating the target position: the difference between the left and right neuron output
        data=(-50.0 * voltage_right + 50.0 * voltage_left )
        #Setting the limit for the joint position (makes sure the mouse head only turns enough to
        # center the red screen, this serves a different function than joint limits in the sdf)
        if abs(data)>0.3:
                    sign=data/(abs(data))# + or -
                    data=0.3*sign
        return std_msgs.msg.Float64(data)
#


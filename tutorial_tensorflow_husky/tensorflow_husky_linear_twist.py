"""
This module contains the transfer function which is responsible for determining the linear twist
component of the husky's movement based on the left and right wheel neuron
"""
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.MapVariable("state", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapSpikeSink("left_wheel_neuron", nrp.brain.actors[1], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("right_wheel_neuron", nrp.brain.actors[2], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def linear_twist(t, state, left_wheel_neuron, right_wheel_neuron):

    # determine the stimulus, only drive the wheels if actively searching
    left = left_wheel_neuron.voltage if state.value == 'searching' else 0.0
    right = right_wheel_neuron.voltage if state.value == 'searching' else 0.0

    # construct and return the movement message
    return geometry_msgs.msg.Twist(
        linear=geometry_msgs.msg.Vector3(x=20.0 * max(left, right), y=0.0, z=0.0),
        angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=250.0 * (right - left)))

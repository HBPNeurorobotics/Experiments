"""
This module contains the transfer function which is responsible for determining the linear twist
component of the husky's movement based on the left and right wheel neuron
"""
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.MapSpikeSink("wheel_neurons", nrp.brain.actors, nrp.raw_signal)
@nrp.Neuron2Robot(Topic('/husky/cmd_vel', geometry_msgs.msg.Twist))
def linear_twist(t, wheel_neurons):
    """
    The transfer function which calculates the linear twist of the husky robot based on the
    voltage of left and right wheel neuron.

    :param t: the current simulation time
    :param left_wheel_neuron: the left wheel neuron device
    :param right_wheel_neuron: the right wheel neuron device
    :return: a geometry_msgs/Twist message setting the linear twist fo the husky robot movement.
    """
    return geometry_msgs.msg.Twist(
        linear=geometry_msgs.msg.Vector3(x=10.0 * min((abs(wheel_neurons.value[0]), abs(wheel_neurons.value[1]))), y=0.0,
                                         z=0.0),
        angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=7.0 * (
            wheel_neurons.value[1] - wheel_neurons.value[0])))

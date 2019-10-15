@nrp.MapSpikeSink("left_wheel_neuron", nrp.brain.left_wheel_motor, nrp.leaky_integrator_exp, weight=0.1)
@nrp.MapSpikeSink("right_wheel_neuron", nrp.brain.right_wheel_motor, nrp.leaky_integrator_exp, weight=0.1)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def linear_twist(t, left_wheel_neuron, right_wheel_neuron):
    return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(x=20.0 * min(left_wheel_neuron.voltage, right_wheel_neuron.voltage), y=0.0, z=0.0), angular=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=100.0 * (right_wheel_neuron.voltage - left_wheel_neuron.voltage)))


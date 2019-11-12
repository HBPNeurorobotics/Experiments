@nrp.MapSpikeSink("left_wheel_neuron", nrp.brain.actors[1], nrp.population_rate)
@nrp.MapSpikeSink("right_wheel_neuron", nrp.brain.actors[2], nrp.population_rate)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def linear_twist_pop_rate(t, left_wheel_neuron, right_wheel_neuron):
    return geometry_msgs.msg.Twist(
        linear=geometry_msgs.msg.Vector3(
            x=0.01 * min(left_wheel_neuron.rate, right_wheel_neuron.rate), y=0.0, z=0.0),
        angular=geometry_msgs.msg.Vector3(
            x=0.0, y=0.0, z=0.05 * (right_wheel_neuron.rate - left_wheel_neuron.rate)))
# Imported Python Transfer Function
#
import numpy as np
@nrp.MapVariable("topic_index", initial_value=-1)
@nrp.MapRobotSubscriber("topic_joint_states", Topic('/joint_states', sensor_msgs.msg.JointState))
@nrp.MapSpikeSource("input_layer", nrp.map_neurons(range(0, 90), lambda i: nrp.brain.sensors[i]), nrp.poisson, weight=100.0)
@nrp.Robot2Neuron()
def propagate_input_signal(t, topic_index, topic_joint_states, input_layer):
    try:
        if not topic_joint_states.value:
            clientLogger.info("ROS topic /joint_states not found, skipping TF: propagate_input_signal")
            return
        def scale(x, L, input_range):
            return float(x - input_range[0]) * L / (input_range[1] - input_range[0])
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        def scalar_encoder(x, length, input_range, std_dev):
            result = np.empty(length, dtype=float)
            for i in xrange(length):
                result[i] = gaussian(i, scale(x, length, input_range), std_dev * length)
            return result
        if topic_index.value == -1:
            topic_index.value = topic_joint_states.value.name.index('hollie_real_left_arm_2_joint')
        joint_value = topic_joint_states.value.position[topic_index.value]
        # encode
        frequencies = scalar_encoder(joint_value, 90, [0, 3.14/2.0], 0.05) * 20.0
        # set rates
        input_layer.rate = frequencies
    except Exception as e:
        clientLogger.info(str(e))
#


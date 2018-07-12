# Imported Python Transfer Function
#
@nrp.MapVariable("topic_index", initial_value=-1)
@nrp.MapVariable("spike_rate_scale", initial_value=125.0)
@nrp.MapSpikeSource("extension", nrp.brain.index_distal_extension_motor, nrp.fixed_frequency)
@nrp.MapSpikeSource("contraction", nrp.brain.index_distal_contraction_motor, nrp.fixed_frequency)
@nrp.MapRobotSubscriber("target_angle", Topic('/target_angle_index_finger_distal', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("topic_joint_state", Topic('/joint_states', sensor_msgs.msg.JointState))
@nrp.Robot2Neuron()
def distal_angle_to_motor(t, topic_index, spike_rate_scale, extension, contraction, target_angle, topic_joint_state):
    try:
        # Fetch current angle value from Gazebo
        if not topic_joint_state.value is None and not target_angle.value is None:
            # Safe repeated linear list search
            if topic_index.value == -1:
                topic_index.value = topic_joint_state.value.name.index('hollie_real_left_hand_Index_Finger_Distal')
            current_angle = topic_joint_state.value.position[topic_index.value]
            # Encode motor neuron activation dynamics
            extension.rate = max(current_angle-target_angle.value.data, 0) * spike_rate_scale.value
            contraction.rate = max(target_angle.value.data-current_angle, 0) * spike_rate_scale.value
    except Exception as e:
        clientLogger.info(str(e))
#


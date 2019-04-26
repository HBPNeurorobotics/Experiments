# Imported Python Transfer Function
#
@nrp.MapRobotSubscriber("joint_data", Topic('/arm_robot/joint_states', sensor_msgs.msg.JointState))
@nrp.MapSpikeSource("shoulder_encoding", nrp.brain.shoulder, nrp.fixed_frequency)
@nrp.MapSpikeSource("elbow_flexion_encoding", nrp.brain.elbow_flexion, nrp.fixed_frequency)
@nrp.MapSpikeSource("elbow_rotation_encoding", nrp.brain.elbow_rotation, nrp.fixed_frequency)
@nrp.Robot2Neuron()
def arm_proprioception(t, joint_data, shoulder_encoding, elbow_flexion_encoding, elbow_rotation_encoding):
    # shoulder encoding
    if joint_data is None:
        return
    shoulder_idx = joint_data.value.name.index("hollie_real_left_arm_4_joint")
    encoded_value = abs(joint_data.value.position[shoulder_idx]) * 20
    shoulder_encoding.rate = encoded_value
    elbow_flexion_idx = joint_data.value.name.index("hollie_real_left_arm_3_joint")
    encoded_value = abs(joint_data.value.position[elbow_flexion_idx]) * 20
    elbow_flexion_encoding.rate = encoded_value
    elbow_rotation_idx = joint_data.value.name.index("hollie_real_left_arm_1_joint")
    encoded_value = abs(joint_data.value.position[elbow_rotation_idx]) * 20
    elbow_rotation_encoding.rate = encoded_value
#


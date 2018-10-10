import gazebo_msgs.msg

@nrp.MapRobotSubscriber("pose", Topic("/gazebo/model_states", gazebo_msgs.msg.ModelStates))
@nrp.MapCSVRecorder("recorder", "all_joints_positions.csv", ["Name", "time", "Position"])
@nrp.Robot2Neuron()
def csv_joint_state_monitor(t, pose, recorder):
    if not isinstance(pose.value, type(None)):
        recorder.record_entry("pose", t, pose.value.pose[0].position.z)
    return
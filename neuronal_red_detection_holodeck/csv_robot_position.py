@nrp.MapCSVRecorder("recorder", filename="robot_position.csv", headers=["x", "y", "z"])
@nrp.MapRobotSubscriber("position", Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapVariable("robot_index", global_key="robot_index", initial_value=None)
@nrp.Robot2Neuron()
def csv_robot_position(t, position, recorder, robot_index):
    if not isinstance(position.value, type(None)):

        # determine if previously set robot index has changed
        if robot_index.value is not None:

            # if the value is invalid, reset the index below
            if robot_index.value >= len(position.value.name) or\
               position.value.name[robot_index.value] != 'robot':
                robot_index.value = None

        # robot index is invalid, find and set it
        if robot_index.value is None:

            # 'robot' is guaranteed by the NRP, if not found raise error
            robot_index.value = position.value.name.index('robot')

        # record the current robot position
        recorder.record_entry(position.value.pose[robot_index.value].position.x,
                              position.value.pose[robot_index.value].position.y,
                              position.value.pose[robot_index.value].position.z)

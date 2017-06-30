@nrp.MapCSVRecorder("recorder", filename="robot_position.csv", headers=["x", "y", "z"])
@nrp.MapRobotSubscriber("position", Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.Robot2Neuron()
def csv_robot_position(t, position, recorder):
    recorder.record_entry(position.value.pose[-1].position.x,
                          position.value.pose[-1].position.y,
                          position.value.pose[-1].position.z)

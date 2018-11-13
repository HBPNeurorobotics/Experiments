@nrp.MapRobotPublisher('r_shoulder_roll', Topic('/icub/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_pitch', Topic('/icub/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_elbow', Topic('/icub/r_elbow/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('eye_tilt', Topic('/icub/eye_tilt/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def simple_move_robot(t, r_shoulder_roll, r_shoulder_pitch , r_elbow, eye_tilt):
    #################################################
    # Insert code here:
    # Move the robot to put its hand in front of the thrown balls
    # You can try to use different control topic
    # All topics can be listed with rostopic list in a terminal - see ROS
    #################################################
    eye_tilt.send_message(std_msgs.msg.Float64(-0.8))

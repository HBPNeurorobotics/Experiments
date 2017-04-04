# Imported Python Transfer Function
@nrp.MapRobotPublisher('r_shoulder_roll', Topic('/iCub/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_pitch', Topic('/iCub/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_yaw', Topic('/iCub/r_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_elbow', Topic('/iCub/r_elbow/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def move_icub(t, r_shoulder_roll, r_shoulder_pitch, r_shoulder_yaw, r_elbow):
    if t < 2.:
        return
    if t % 1. < 0.5:
        elbow = 0.
    else:
        elbow = 1.5
    r_shoulder_roll.send_message(std_msgs.msg.Float64(1.7))
    r_shoulder_pitch.send_message(std_msgs.msg.Float64(-1.1))
    r_shoulder_yaw.send_message(std_msgs.msg.Float64(0.))
    r_elbow.send_message(std_msgs.msg.Float64(elbow))

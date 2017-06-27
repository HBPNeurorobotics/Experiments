# Imported Python Transfer Function
@nrp.MapRobotPublisher('l_shoulder_roll', Topic('/iCub/l_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_pitch', Topic('/iCub/l_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_yaw', Topic('/iCub/l_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_elbow', Topic('/iCub/l_elbow/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_roll', Topic('/iCub/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_pitch', Topic('/iCub/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_yaw', Topic('/iCub/r_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_elbow', Topic('/iCub/r_elbow/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def move_insider_icub(t,
    l_shoulder_roll, l_shoulder_pitch, l_shoulder_yaw, l_elbow, 
    r_shoulder_roll, r_shoulder_pitch, r_shoulder_yaw, r_elbow):

    def wave_hand(side, roll, pitch, yaw, elbow):
        if side == 1:
            r_shoulder_roll.send_message(std_msgs.msg.Float64(roll))
            r_shoulder_pitch.send_message(std_msgs.msg.Float64(pitch))
            r_shoulder_yaw.send_message(std_msgs.msg.Float64(yaw))
            r_elbow.send_message(std_msgs.msg.Float64(elbow))
        elif side == -1:
            l_shoulder_roll.send_message(std_msgs.msg.Float64(roll))
            l_shoulder_pitch.send_message(std_msgs.msg.Float64(pitch))
            l_shoulder_yaw.send_message(std_msgs.msg.Float64(yaw))
            l_elbow.send_message(std_msgs.msg.Float64(elbow))

    if t < 2.:
        return
    if t % 1. < 0.5:
        elbow = 0.
    else:
        elbow = 1.5

    if t % 20 < 10.:
        wave_hand(1, 0., 0., 0., 0.)
        wave_hand(-1, 1.7, -1.1, 0., elbow)
    else:
        wave_hand(-1, 0., 0., 0., 0.)
        wave_hand(1, 1.7, -1.1, 0., elbow)


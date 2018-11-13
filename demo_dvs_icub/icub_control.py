# Imported Python Transfer Function
import numpy as np
from dvs_msgs.msg import EventArray

# Imported Python Transfer Function
@nrp.MapVariable("last_event_count", initial_value=0)
@nrp.MapRobotSubscriber('notification', Topic('/robot/notifications', std_msgs.msg.String))
@nrp.MapRobotSubscriber('dvs_narrow_view', Topic('/dvs_narrow_view', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher('l_shoulder_roll', Topic('/robot/l_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_pitch', Topic('/robot/l_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_yaw', Topic('/robot/l_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_elbow', Topic('/robot/l_elbow/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_roll', Topic('/robot/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_pitch', Topic('/robot/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_yaw', Topic('/robot/r_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_elbow', Topic('/robot/r_elbow/pos', std_msgs.msg.Float64))
@nrp.Neuron2Robot()
def icub_control(t, last_event_count, notification, dvs_narrow_view, 
    l_shoulder_roll, l_shoulder_pitch, l_shoulder_yaw, l_elbow, 
    r_shoulder_roll, r_shoulder_pitch, r_shoulder_yaw, r_elbow):
    
    def wave_hand(time, side):
        if time % 1. < 0.5:
            elbow = 0.
        else:
            elbow = 1.5
        # wave the left hand
        if side == -1:
            l_shoulder_roll.send_message(std_msgs.msg.Float64(1.7))
            l_shoulder_pitch.send_message(std_msgs.msg.Float64(-1.1))
            l_shoulder_yaw.send_message(std_msgs.msg.Float64(0.))
            l_elbow.send_message(std_msgs.msg.Float64(elbow))
        # wave the right hand
        elif side == 1:
            r_shoulder_roll.send_message(std_msgs.msg.Float64(1.7))
            r_shoulder_pitch.send_message(std_msgs.msg.Float64(-1.1))
            r_shoulder_yaw.send_message(std_msgs.msg.Float64(0.))
            r_elbow.send_message(std_msgs.msg.Float64(elbow))


    narrow_view_msg = dvs_narrow_view.value
    if narrow_view_msg is None or t < 4.:
        return

    event_count = 0
    height = narrow_view_msg.height
    width = narrow_view_msg.width
    half_width = width/2
    dvs_image = CvBridge().imgmsg_to_cv2(narrow_view_msg, 'rgb8')
    for i in range(width):
        for j in range(height):
            if dvs_image[j][i][0] != 0:
                if i < half_width - 2:
                    event_count -= 1
                elif i > half_width + 2:
                    event_count += 1

    threshold = 33
    side = -1
    if notification.value != None and notification.value.data == "switch":
        side = 1
    if last_event_count.value < -threshold and event_count < -threshold:
        wave_hand(t, side)
    if last_event_count.value > threshold and event_count > threshold:
        wave_hand(t, - side)

    last_event_count.value = event_count
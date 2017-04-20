# Imported Python Transfer Function
from dvs_msgs.msg import EventArray
import numpy as np
@nrp.MapRobotPublisher('dvs_rendered', Topic('/dvs_rendered_full', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("dvs", Topic('head/dvs_left/events', EventArray))
@nrp.Robot2Neuron()
def display_events(t, dvs, dvs_rendered):
    event_msg = dvs.value
    if event_msg is None:
        return
    rendered_img = np.zeros((128, 128, 3), dtype=np.uint8)
    for event in event_msg.events:
        rendered_img[event.y][event.x] = (event.polarity * 255, 255, 0)
    msg_frame = CvBridge().cv2_to_imgmsg(rendered_img, 'rgb8')
    dvs_rendered.send_message(msg_frame)

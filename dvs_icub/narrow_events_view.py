# Imported Python Transfer Function
"""
Narrow window surrounding the waving hands of the insder iCub
"""
from dvs_msgs.msg import EventArray
import numpy as np
@nrp.MapRobotPublisher('dvs_narrow_view', Topic('/dvs_narrow_view', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("dvs", Topic('head/dvs_left/events', EventArray))
@nrp.Robot2Neuron()
def narrow_events_view(t, dvs, dvs_narrow_view):
    event_msg = dvs.value
    if event_msg is None:
        return
    # There are too many events - we randomly select a subset of them
    n_events_to_keep = min(450, len(event_msg.events))
    filtered_events = np.random.choice(event_msg.events, n_events_to_keep, replace=False)
    top = 55
    bottom = 22
    eye_offset = 2
    middle = 64
    center = middle + eye_offset
    radius = 40
    left = center - radius
    right = center + radius
    rendered_img = np.zeros((top - bottom, right - left, 3), dtype=np.uint8)
    for event in filtered_events:
        if event.y < top and event.y >= bottom and event.x < right and event.x >= left:
                rendered_img[event.y - bottom][event.x - left] = (event.polarity * 255, 255, 0)

    for j in range(top - bottom):        
      rendered_img[j - bottom][radius] = (255, 0, 0)
    msg_frame = CvBridge().cv2_to_imgmsg(rendered_img, 'rgb8')
    dvs_narrow_view.send_message(msg_frame)

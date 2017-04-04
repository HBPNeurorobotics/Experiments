# Imported Python Transfer Function
"""
This module contains the transfer function that transforms DVS address events
to spikes and input them to a population of neurons
"""
from dvs_msgs.msg import EventArray
import numpy as np
@nrp.MapRobotPublisher('dvs_rendered', Topic('/dvs_rendered', sensor_msgs.msg.Image))
@nrp.MapRobotSubscriber("dvs", Topic('head/dvs_left/events', EventArray))
@nrp.MapSpikeSource("input_neurons", nrp.map_neurons(
    range(0,nrp.config.brain_root.input_shape[0] * nrp.config.brain_root.input_shape[1]),
    lambda i: nrp.brain.sensors[i]), nrp.dc_source)
@nrp.Robot2Neuron()
def grab_events(t, dvs, input_neurons, dvs_rendered):
    event_msg = dvs.value
    amplitudes = np.zeros(nrp.config.brain_root.input_shape[0] * nrp.config.brain_root.input_shape[1])
    if event_msg is None:
        input_neurons.amplitude = amplitudes
        return

    # There are too many events - we randomly select a subset of them
    n_events_to_keep = min(100, len(event_msg.events))
    filtered_events = np.random.choice(event_msg.events, n_events_to_keep, replace=False)
    rendered_img = np.zeros((nrp.config.brain_root.input_shape[0], nrp.config.brain_root.input_shape[1], 3), dtype=np.uint8)
    # set high amplitude for neurons that spiked
    for event in filtered_events:
        rescaled_event = (np.array((event.y, event.x)) * nrp.config.brain_root.scaling_factor).astype(int)
        rendered_img[rescaled_event[0]][rescaled_event[1]] = (event.polarity * 255, 255, 0)
        idx = rescaled_event[0] * nrp.config.brain_root.input_shape[1] + rescaled_event[1]
        amplitudes[idx] = 1
    input_neurons.amplitude = amplitudes
    #active_neurons = input_neurons[active_neurons].amplitude = 1
    msg_frame = CvBridge().cv2_to_imgmsg(rendered_img, 'rgb8')
    dvs_rendered.send_message(msg_frame)

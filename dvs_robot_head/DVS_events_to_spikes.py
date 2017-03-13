"""
This module contains the transfer function that transforms DVS address events
to spikes and input them to a population of neurons
"""
from dvs_msgs.msg import EventArray
import numpy as np

@nrp.MapVariable("scaling_factor", initial_value=0.1, scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("dvs", Topic('head/dvs_left/events', EventArray))
@nrp.MapSpikeSource("input_neurons", nrp.map_neurons(range(0, 13*13), lambda i: nrp.brain.sensors[i]), nrp.dc_source)
@nrp.Robot2Neuron()
def grab_events(t, dvs, input_neurons, scaling_factor):
    event_msg = dvs.value
    if event_msg is None:
        return

    # reset all amplitudes
    for n in input_neurons:
        n.amplitude = 0

    # There are too many events - we randomly select a subset of them
    n_events_to_keep = min(30, len(event_msg.events))
    filtered_events = np.random.choice(event_msg.events, n_events_to_keep, replace=False)

    # set high amplitude for neurons that spiked
    for event in filtered_events:
        rescaled_event = (np.array((event.x, event.y)) * scaling_factor.value).astype(int)
        idx = rescaled_event[0] * 13 + rescaled_event[1]
        input_neurons[idx].amplitude = 1

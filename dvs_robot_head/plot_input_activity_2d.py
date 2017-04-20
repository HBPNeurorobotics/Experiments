@nrp.MapSpikeSink("sensor_recorder", nrp.brain.sensors, nrp.spike_recorder)
@nrp.MapRobotPublisher('input_layer_plot', Topic('/input_layer', sensor_msgs.msg.Image))
@nrp.Neuron2Robot()
def plot_input_activity_2d (t, input_layer_plot, sensor_recorder):
    # we count the spikes of each motor neurons
    input_shape = np.array(nrp.config.brain_root.input_shape)
    spike_array = np.zeros(input_shape[0] *
                           input_shape[1])
    spike_times = sensor_recorder.times.tolist()
    if len(spike_times) == 0:
        return
    min_idx = int(sensor_recorder.neurons[0])
    for (idx, time) in spike_times:
        idx = idx - min_idx
        spike_array[idx] = spike_array[idx] + 1

    # plot the activity of the output layer in 2D
    input_activity = spike_array.reshape(input_shape[0], input_shape[1])
    input_activity = input_activity * 255. / np.amax(input_activity)
    msg_frame = CvBridge().cv2_to_imgmsg(input_activity.astype(np.uint8), 'mono8')
    input_layer_plot.send_message(msg_frame)

@nrp.MapCSVRecorder("recorder", filename="all_spikes.csv", headers=["id", "time"])
@nrp.MapSpikeSink("red_left_eye", nrp.brain.red_left_eye, nrp.spike_recorder)
#@nrp.MapSpikeSink("red_right_eye", nrp.brain.red_right_eye, nrp.spike_recorder)
#@nrp.MapSpikeSink("green_blue_eye", nrp.brain.green_blue_eye, nrp.spike_recorder)
#@nrp.MapSpikeSink("go_on", nrp.brain.go_on, nrp.spike_recorder)
#@nrp.MapSpikeSink("left_wheel_motor", nrp.brain.left_wheel_motor, nrp.spike_recorder)
#@nrp.MapSpikeSink("right_wheel_motor", nrp.brain.right_wheel_motor, nrp.spike_recorder)
@nrp.Neuron2Robot(Topic('/monitor/spike_recorder', cle_ros_msgs.msg.SpikeEvent))
def csv_spike_monitor(t, recorder, red_left_eye):#, red_right_eye, green_blue_eye, go_on, left_wheel_motor, right_wheel_motor):
    to_record = [(red_left_eye, 2)]#, (red_right_eye, 2), (green_blue_eye, 1), (go_on, 1), (left_wheel_motor, 1), (right_wheel_motor, 1)]
    offset = 0
    for record_neurons, n_neurons in to_record:
        for i in range(0, len(record_neurons.times)):
            recorder.record_entry(
                record_neurons.times[i][0] + offset,
                record_neurons.times[i][1]
            )
    offset += n_neurons

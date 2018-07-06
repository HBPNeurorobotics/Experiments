    # Imported Python Transfer Function
    #
    from std_msgs.msg import Float64
    @nrp.MapVariable("eye_velocity", scope=nrp.GLOBAL)
    @nrp.MapVariable("eye_position", scope=nrp.GLOBAL)
    @nrp.MapVariable("lastballpos", initial_value=-1)
    @nrp.MapSpikeSink("ganglion", nrp.brain.circuit[slice(0, 1280, 1)], nrp.spike_recorder)
    @nrp.MapVariable("trajectory", initial_value="sinusoidal", scope=nrp.GLOBAL)
    @nrp.Neuron2Robot(Topic('/robot/eye_version/pos', Float64))
    def move_eye(t, eye_velocity, eye_position, lastballpos, ganglion, trajectory):
        deg2rad = lambda deg: (float(deg) / 360.) * (2. * np.pi)
        tf = hbp_nrp_cle.tf_framework.tf_lib
        spike_counts_OFF = [0]*320
        spike_counts_ON = [0]*320
        for ev in ganglion.times:
            if ev[0] > 640 and ev[0] <= 960:
                idx = int(ev[0]-641)
                spike_counts_OFF[idx] = spike_counts_OFF[idx] + 1
            elif ev[0] > 960 and ev[0] <= 1280:
                idx = int(ev[0]-961)
                spike_counts_ON[idx] = spike_counts_ON[idx] + 1
        highest_spiking_count_OFF = max(spike_counts_OFF)
        highest_spiking_count_ON = max(spike_counts_ON)
        most_spiking_neuron_OFF = spike_counts_OFF.index(highest_spiking_count_OFF)
        most_spiking_neuron_ON = spike_counts_ON.index(highest_spiking_count_ON)
        # This is the current ball position estimation
        ball_pos = (most_spiking_neuron_ON + most_spiking_neuron_OFF) / 2.
        # Waiting some time for the robot to stabilize
        if t < 2:
            return eye_position.value
        if highest_spiking_count_OFF == 0 or highest_spiking_count_ON == 0:
            return eye_position.value
        if abs(ball_pos - lastballpos.value) > 20 and lastballpos.value != -1.:
            return eye_position.value
        ret = tf.cam.pixel2angle(ball_pos, 0)[0]
        lastballpos.value = ball_pos
        k = 0.2
        return eye_position.value + k*deg2rad(ret)
    #

    # Imported Python Transfer Function
    #
    @nrp.MapSpikeSink("motor_neuron", nrp.brain.motor, nrp.spike_recorder)
    @nrp.MapVariable("index_proximal", initial_value=0.0)
    @nrp.MapVariable("index_distal", initial_value=0.0)
    @nrp.MapVariable("middle_proximal", initial_value=0.0)
    @nrp.MapVariable("middle_distal", initial_value=0.0)
    @nrp.MapVariable("ring_proximal", initial_value=0.0)
    @nrp.MapVariable("ring_distal", initial_value=0.0)
    @nrp.MapVariable("pinky_distal", initial_value=0.0)
    @nrp.MapVariable("pinky_proximal", initial_value=0.0)
    @nrp.MapVariable("thumb_flexion", initial_value=0.0)
    @nrp.MapVariable("thumb_distal", initial_value=0.0)
    @nrp.MapRobotPublisher("topic_index_proximal", Topic('/robot/hollie_real_left_hand_Index_Finger_Proximal/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_index_distal", Topic('/robot/hollie_real_left_hand_Index_Finger_Distal/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_middle_proximal", Topic('/robot/hollie_real_left_hand_Middle_Finger_Proximal/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_middle_distal", Topic('/robot/hollie_real_left_hand_Middle_Finger_Distal/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_ring_proximal", Topic('/robot/hollie_real_left_hand_Ring_Finger/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_ring_distal", Topic('/robot/hollie_real_left_hand_j12/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_pinky_proximal", Topic('/robot/hollie_real_left_hand_Pinky/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_pinky_distal", Topic('/robot/hollie_real_left_hand_j13/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_thumb_flexion", Topic('/robot/hollie_real_left_hand_j4/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotPublisher("topic_thumb_distal", Topic('/robot/hollie_real_left_hand_j3/cmd_pos', std_msgs.msg.Float64))
    @nrp.Neuron2Robot()
    def muscle_model_index_proximal(t, motor_neuron,
                                    index_proximal, topic_index_proximal,
                                    index_distal, topic_index_distal,
                                    middle_proximal, topic_middle_proximal,
                                    middle_distal, topic_middle_distal,
                                    ring_proximal, topic_ring_proximal,
                                    ring_distal, topic_ring_distal,
                                    pinky_proximal, topic_pinky_proximal,
                                    pinky_distal, topic_pinky_distal,
                                    thumb_flexion, topic_thumb_flexion,
                                    thumb_distal, topic_thumb_distal):
        motion = 0.03
        if motor_neuron.spiked:
            spiked = motion
        else:
            spiked = - motion * 2
        index_proximal.value = np.clip(index_proximal.value + spiked, 0, 1.2)
        topic_index_proximal.send_message(std_msgs.msg.Float64(index_proximal.value))
        index_distal.value = np.clip(index_distal.value + spiked, 0, 1.5)
        topic_index_distal.send_message(std_msgs.msg.Float64(index_distal.value))
        middle_proximal.value = np.clip(middle_proximal.value + spiked, 0, 1.2)
        topic_middle_proximal.send_message(std_msgs.msg.Float64(middle_proximal.value))
        middle_distal.value = np.clip(middle_distal.value + spiked, 0, 1.5)
        topic_middle_distal.send_message(std_msgs.msg.Float64(middle_distal.value))
        ring_proximal.value = np.clip(ring_proximal.value + spiked, 0, 1.2)
        topic_ring_proximal.send_message(std_msgs.msg.Float64(ring_proximal.value))
        ring_distal.value = np.clip(ring_distal.value + spiked, 0, 1.5)
        topic_ring_distal.send_message(std_msgs.msg.Float64(ring_distal.value))
        pinky_proximal.value = np.clip(pinky_proximal.value + spiked, 0, 1.2)
        topic_pinky_proximal.send_message(std_msgs.msg.Float64(pinky_proximal.value))
        pinky_distal.value = np.clip(pinky_distal.value + spiked, 0, 1.2)
        topic_pinky_distal.send_message(std_msgs.msg.Float64(pinky_distal.value))
        thumb_flexion.value = np.clip(thumb_flexion.value - spiked, -0.5, 0.5)
        topic_thumb_flexion.send_message(std_msgs.msg.Float64(thumb_flexion.value))
        thumb_distal.value = np.clip(thumb_distal.value - spiked, -0.5, 0.5)
        topic_thumb_distal.send_message(std_msgs.msg.Float64(thumb_distal.value))
        #

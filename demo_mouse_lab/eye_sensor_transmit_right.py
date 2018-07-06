    # Imported Python Transfer Function
    #
    import sensor_msgs.msg
    import hbp_nrp_cle.tf_framework.tf_lib
    @nrp.MapRobotSubscriber("camera_left", Topic('/mouse_left_eye/mouse/left_eye', sensor_msgs.msg.Image))
    @nrp.MapRobotSubscriber("camera_right", Topic('/mouse_right_eye/mouse/right_eye', sensor_msgs.msg.Image))
    @nrp.MapSpikeSource("red_left_eye", nrp.brain.sensors[0], nrp.poisson)
    @nrp.MapSpikeSource("red_right_eye", nrp.brain.sensors[1], nrp.poisson)
    @nrp.Robot2Neuron()
    def eye_sensor_transmit_right(t, camera_right, camera_left, red_left_eye, red_right_eye):
        image_results_left = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera_left.value)
        image_results_right = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera_right.value)
        red_right_eye.rate = 500.0 * (image_results_left.right+image_results_right.right)
        red_left_eye.rate = 500.0 * (image_results_left.left+image_results_right.left)
    #

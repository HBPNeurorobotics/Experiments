    # Imported Python Transfer Function
    #
    @nrp.MapRobotSubscriber("camera", Topic('/husky/camera', sensor_msgs.msg.Image))
    @nrp.MapSpikeSource("red_left_eye", nrp.map_neurons(range(0, 300), lambda i: nrp.brain.sensors[i]), nrp.ac_source)
    @nrp.MapSpikeSource("red_right_eye", nrp.map_neurons(range(300, 600), lambda i: nrp.brain.sensors[i]), nrp.ac_source)
    @nrp.MapSpikeSource("green_blue_eyes", nrp.brain.circuit[708], nrp.poisson)
    @nrp.Robot2Neuron()
    def eye_sensor_transmit(t, camera, red_left_eye, red_right_eye, green_blue_eyes):
        image_results = hbp_nrp_cle.tf_framework.tf_lib.get_color_values(image=camera.value, width=30, height=20)
        red_left_eye.amplitude = 1.0 * image_results.left_red -\
                                 0.5 * image_results.left_blue -\
                                 0.5 * image_results.left_green
        red_right_eye.amplitude = 1.0 * image_results.right_red -\
                                  0.5 * image_results.right_blue -\
                                  0.5 * image_results.right_green
        green_blue_eyes.rate = 1000.0 * np.mean(image_results.right_blue + image_results.left_blue +\
                                                image_results.right_green + image_results.left_green)
    #

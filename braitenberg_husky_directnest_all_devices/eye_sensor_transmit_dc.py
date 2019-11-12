@nrp.MapRobotSubscriber("camera", Topic('/husky/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_left_eye", nrp.brain.sensors[slice(0, 3, 2)], nrp.dc_source)
@nrp.MapSpikeSource("red_right_eye", nrp.brain.sensors[slice(1, 4, 2)], nrp.dc_source)
@nrp.MapSpikeSource("green_blue_eye", nrp.brain.sensors[4], nrp.dc_source)
@nrp.Robot2Neuron()
def eye_sensor_transmit_dc(t, camera, red_left_eye, red_right_eye, green_blue_eye):
    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    red_left_eye.amplitude = 200.0 * image_results.left
    red_right_eye.amplitude = 200.0 * image_results.right
    green_blue_eye.amplitude = 7.5 * image_results.go_on

@nrp.MapRobotSubscriber("camera", Topic('/husky/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_left_eye", nrp.brain.red_left_eye, nrp.poisson)
@nrp.MapSpikeSource("red_right_eye", nrp.brain.red_right_eye, nrp.poisson)
@nrp.MapSpikeSource("green_blue_eye", nrp.brain.green_blue_eye, nrp.poisson)
@nrp.Robot2Neuron()
def eye_sensor_transmit(t, camera, red_left_eye, red_right_eye, green_blue_eye):
    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    red_left_eye.rate = 2000.0 * image_results.left
    red_right_eye.rate = 2000.0 * image_results.right
    green_blue_eye.rate = 75.0 * image_results.go_on


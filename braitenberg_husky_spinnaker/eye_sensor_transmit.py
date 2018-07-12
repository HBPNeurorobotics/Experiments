# Imported Python Transfer Function
#
import sensor_msgs.msg
import hbp_nrp_cle.tf_framework.tf_lib #import detect_red
@nrp.MapRobotSubscriber("camera", Topic('/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_left_eye1", nrp.brain.circuit[0], nrp.poisson, delay=0.1, weight=5.0)
@nrp.MapSpikeSource("red_left_eye2", nrp.brain.circuit[2], nrp.poisson, delay=0.1, weight=5.0)
@nrp.MapSpikeSource("red_right_eye1", nrp.brain.circuit[1], nrp.poisson, delay=0.1, weight=5.0)
@nrp.MapSpikeSource("red_right_eye2", nrp.brain.circuit[3], nrp.poisson, delay=0.1, weight=5.0)
@nrp.MapSpikeSource("green_blue_eye", nrp.brain.circuit[4], nrp.poisson, delay=0.1, weight=5.0)
@nrp.Robot2Neuron()
def eye_sensor_transmit(t, camera, red_left_eye1, red_left_eye2, red_right_eye1, red_right_eye2, green_blue_eye):
    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    red_left_eye1.rate = 1000.0 * image_results.left
    red_left_eye2.rate = 1000.0 * image_results.left
    red_right_eye1.rate = 1000.0 * image_results.right
    red_right_eye2.rate = 1000.0 * image_results.right
    green_blue_eye.rate = 1000.0 * image_results.go_on
#


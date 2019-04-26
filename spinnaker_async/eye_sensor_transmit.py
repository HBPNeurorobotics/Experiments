# Imported Python Transfer Function
import sensor_msgs.msg
import hbp_nrp_cle.tf_framework.tf_lib #import detect_red
@nrp.MapRobotSubscriber("camera", Topic('/husky/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("left_injector", nrp.brain.left, nrp.injector, n=10)
@nrp.MapSpikeSource("right_injector", nrp.brain.right, nrp.injector, n=10)
@nrp.MapVariable("last", initial_value=(True, True))
@nrp.Robot2Neuron(triggers="camera")
def eye_sensor_transmit(t, camera, left_injector, right_injector, last):
    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    found_left = False
    found_right = False
    if image_results.left * 10 > image_results.go_on:
        found_left = True
        if not last.value[0]:
            clientLogger.info("Found red color left")
        right_injector.inject_spikes()
    if image_results.right * 10 > image_results.go_on:
        found_right = True
        if not last.value[1]:
            clientLogger.info("Found red color right")
        left_injector.inject_spikes()
    if not found_left and not found_right:
        if last.value[0] or last.value[1]:
            clientLogger.info("Found no red color")
        right_injector.inject_spikes()
    last.value = (found_left, found_right)
##


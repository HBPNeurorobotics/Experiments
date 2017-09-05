# Imported Python Transfer Function
import numpy as np
import sensor_msgs.msg
from cv_bridge import CvBridge
@nrp.MapRobotSubscriber("camera", Topic("/icub_model/left_eye_camera/image_raw", sensor_msgs.msg.Image))
@nrp.MapSpikeSource("sensors", nrp.map_neurons(range(0, nrp.config.brain_root.n_sensors), lambda i: nrp.brain.sensors[i]), nrp.dc_source)
@nrp.MapVariable("last_mean_green", initial_value=None, scope=nrp.GLOBAL)
@nrp.Robot2Neuron()
def grab_image(t, camera, sensors, last_mean_green):
    image_msg = camera.value
    if image_msg is not None:
        img = CvBridge().imgmsg_to_cv2(image_msg, "rgb8")
        # img is a numpy array representing an OpenCV RGB 8 image

        ####################################################################################################
        # Insert code here: change this line to get the green pixles from the cv image (this is easily googleable)
        ####################################################################################################
        mean_green = 0.

        # You can use last_mean_green variable for persistenty accross calls
        if last_mean_green.value is None:
            last_mean_green.value = mean_green
        delta_mean_green = mean_green - last_mean_green.value

        # We now set the amplitude of the input neurons
        for neuron in sensors:
            neuron.amplitude = 3. * max(0., delta_mean_green) * np.random.rand()
        last_mean_green.value = mean_green

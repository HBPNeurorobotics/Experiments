# Imported Python Transfer Function
#
@nrp.MapRobotSubscriber("camera", Topic('/husky/camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("red_left_eye", nrp.brain.sensors[slice(0, 3, 2)], nrp.poisson)
@nrp.MapSpikeSource("red_right_eye", nrp.brain.sensors[slice(1, 4, 2)], nrp.poisson)
@nrp.MapSpikeSource("green_blue_eye", nrp.brain.sensors[4], nrp.poisson)
@nrp.Robot2Neuron()
def eye_sensor_transmit(t, camera, red_left_eye, red_right_eye, green_blue_eye):
    """
    This transfer function uses OpenCV to compute the percentages of red pixels
    seen by the robot on his left and on his right. Then, it maps these percentages
    (see decorators) to the neural network neurons using a Poisson generator.
    """
    bridge = CvBridge()
    red_left = red_right = green_blue = 0.0
    if not isinstance(camera.value, type(None)):
        # Boundary limits of what we consider red (in HSV format)
        lower_red = np.array([0, 30, 30])
        upper_red = np.array([0, 255, 255])
        # Get an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(camera.value, "rgb8")
        # Transform image to HSV (easier to detect colors).
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
        # Create a mask where every non red pixel will be a zero.
        mask = cv2.inRange(hsv_image, lower_red, upper_red)
        image_size = (cv_image.shape[0] * cv_image.shape[1])
        if (image_size > 0):
            # Since we want to get left and right red values, we cut the image
            # in 2.
            half = cv_image.shape[1] // 2
            # Get the number of red pixels in the image.
            red_left = cv2.countNonZero(mask[:, :half])
            red_right = cv2.countNonZero(mask[:, half:])
            # We have to multiply the red rates by 2 since it is for an
            # half image only. We also multiply all of them by 1000 so that
            # we have enough spikes produced by the Poisson generator
            red_left_eye.rate = 2 * 1000 * (red_left / float(image_size))
            red_right_eye.rate = 2 * 1000 * (red_right / float(image_size))
            green_blue_eye.rate = 75 * ((image_size - (red_left + red_right)) / float(image_size))
#


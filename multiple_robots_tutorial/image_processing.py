import hbp_nrp_cle.tf_framework as nrp

@nrp.MapRobotSubscriber("camera", Topic('/icub/icub_model/left_eye_camera/image_raw', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("left_input_neuron", nrp.brain.icub_input_left[0], nrp.poisson)
@nrp.MapSpikeSource("right_input_neuron", nrp.brain.icub_input_right[0], nrp.poisson)
@nrp.Robot2Neuron()
def eye_sensor_transmit(t, camera, left_input_neuron, right_input_neuron):
    """
    This transfer function uses OpenCV to compute the percentages of red and green pixels
    seen by the iCub robot. Then, it maps these percentages
    (see decorators) to the neural network neurons using a Poisson generator.
    """
    bridge = CvBridge()
    red_pixels = green_pixels = 0.0
    if not isinstance(camera.value, type(None)):

        # Boundary limits of what we consider red (in HSV format)
        lower_red = np.array([0, 30, 30])
        upper_red = np.array([0, 255, 255])
        # Boundary limits of what we consider green (in HSV format)
        lower_green = np.array([55, 30, 30])
        upper_green = np.array([65, 255, 255])

        # Get an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(camera.value, "rgb8")

        # Transform image to HSV (easier to detect colors).
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        # Create a mask where every non red pixel will be a zero.
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        image_size = (cv_image.shape[0] * cv_image.shape[1])

        if (image_size > 0):
            # Get the number of red and green pixels in the image.
            red_pixels = cv2.countNonZero(red_mask)
            green_pixels = cv2.countNonZero(green_mask)
            
            # Turns pixel numbers into spike rates
            magnitude = 0.25
            left_input_neuron.rate = magnitude * red_pixels
            right_input_neuron.rate = magnitude * green_pixels
                

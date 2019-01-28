"""
This module contains the transfer function which is responsible for decoding color input into semantic
objects to search for.
"""
@nrp.MapVariable("state", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapVariable("target", initial_value=None, scope=nrp.GLOBAL)
@nrp.MapRobotSubscriber("camera", Topic('/husky/husky/camera', sensor_msgs.msg.Image))
@nrp.Robot2Neuron()
def input_detector(t, state, target, camera):

    # only detect user input if we are starting at the TV
    if state.value != 'waiting' or target.value is not None:
        return

    # use an NRP convenience function to extract RGB color ratios
    image_results = hbp_nrp_cle.tf_framework.tf_lib.get_color_values(image=camera.value, width=30, height=20)

    # determine the average color density in the image
    r = np.mean(image_results.left_red) + np.mean(image_results.right_red)
    g = np.mean(image_results.left_green) + np.mean(image_results.right_green)
    b = np.mean(image_results.left_blue) + np.mean(image_results.right_blue)

    # check if any surpass a basic threshold (percent of total pixels in image)
    highest = max(r, max(g, b))
    if highest < 0.06:
        return

    # semantically identify the target
    if r == highest:
        target.value = 'chair'
    elif g == highest:
        target.value = 'potted plant'
    elif b == highest:
        target.value = 'couch'

    # log the target to be searched for and start searching
    if target is not None:
        clientLogger.info('User input processed, I will find the', target.value)
        state.value = 'searching'

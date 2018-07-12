# Imported Python Transfer Function
#
import sensor_msgs.msg
@nrp.MapRobotSubscriber("camera", Topic('/left_camera/robot_left_camera', sensor_msgs.msg.Image))
@nrp.MapSpikeSource("input_neuron", nrp.brain.neurons[0], nrp.poisson)
@nrp.Robot2Neuron()
# Example TF: get image and fire at constant rate. You could do something with the image here and fire accordingly.
def grab_image(t, camera, input_neuron):
    image = camera.value
    input_neuron.rate = 10
#


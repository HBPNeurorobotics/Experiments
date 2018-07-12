# Imported Python Transfer Function
#
import std_msgs.msg
from rospy.numpy_msg import numpy_msg
@nrp.MapRobotSubscriber("SNL_ganglion_data", Topic('/icub_model/left_eye_camera/retina/SNL_ganglion/data', numpy_msg(std_msgs.msg.Float64MultiArray)))
@nrp.MapSpikeSource("neurons", nrp.map_neurons(range(0, 320), lambda i: nrp.brain.sensors[i]), nrp.dc_source)
@nrp.Robot2Neuron()
# Example TF: get image and fire at constant rate. You could do something with the image here and fire accordingly.
def grab_data(t, neurons, SNL_ganglion_data):
    getValue = hbp_nrp_cle.tf_framework.tf_lib.getValueFromFloat64MultiArray
    data_msg = SNL_ganglion_data.value
    if data_msg is not None:
        magic_row = 70
        for column in xrange(320):
            neurons[column].amplitude = getValue(data_msg, magic_row, column)
#


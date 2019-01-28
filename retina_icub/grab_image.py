# Imported Python Transfer Function
#
import std_msgs.msg
from rospy.numpy_msg import numpy_msg
@nrp.MapRobotSubscriber("ganglion_ON_data",Topic('/icub/icub_model/left_eye_camera/retina/SNL_ganglion_ON/data', numpy_msg(std_msgs.msg.Float64MultiArray)))
@nrp.MapRobotSubscriber("ganglion_OFF_data",Topic('/icub/icub_model/left_eye_camera/retina/SNL_ganglion_OFF/data', numpy_msg(std_msgs.msg.Float64MultiArray)))
@nrp.MapSpikeSource("ganglion_OFF", nrp.map_neurons(range(0, 320), lambda i: nrp.brain.ganglion_input_OFF[i]), nrp.dc_source)
@nrp.MapSpikeSource("ganglion_ON", nrp.map_neurons(range(0, 320), lambda i: nrp.brain.ganglion_input_ON[i]), nrp.dc_source)
@nrp.Robot2Neuron()
def grab_image(t, ganglion_OFF, ganglion_ON, ganglion_ON_data, ganglion_OFF_data):
    getValue = hbp_nrp_cle.tf_framework.tf_lib.getValueFromFloat64MultiArray
    msg_ON = ganglion_ON_data.value
    msg_OFF = ganglion_OFF_data.value
    if msg_ON is not None and msg_OFF is not None :
        # magic_row = 100
        magic_row = 73
        for column in xrange(320):
            ganglion_OFF[column].amplitude = getValue(msg_OFF, magic_row, column)
            ganglion_ON[column].amplitude = getValue(msg_ON, magic_row, column)
#


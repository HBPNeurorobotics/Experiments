# Imported Python Transfer Function
#
from sensor_msgs.msg import JointState
@nrp.MapRobotSubscriber("joints", Topic("/robot/joints", JointState))
@nrp.Neuron2Robot(Topic('/robot/joint_states', JointState))
def joint_states_passthrough(t, joints):
    return joints.value
#


import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from sensor_msgs.msg import JointState
import std_msgs.msg
from std_msgs.msg import Float64
@nrp.MapRobotSubscriber("joints", Topic("/icub/joints", JointState))
@nrp.Neuron2Robot(Topic('/icub/neck_yaw/vel', Float64))
def head_twist (t, joints):
     
    MAX_AMPLITUDE = 0.8
    RELATIVE_AMPLITUDE = 1.0
       
    joints = joints.value
    head_pos = joints.position[joints.name.index('neck_yaw')]
        
    desired_speed=-np.cos(t*2*np.pi)*MAX_AMPLITUDE*RELATIVE_AMPLITUDE*2.0*3.141592
    desired_pos=-np.sin(t*2*np.pi)*MAX_AMPLITUDE*RELATIVE_AMPLITUDE
        
    if desired_speed > 0:
        desired_speed *= (1+(desired_pos - head_pos)*2)
    else:
        desired_speed *= (1-(desired_pos - head_pos)*2)
        
    return std_msgs.msg.Float64(desired_speed)

import hbp_nrp_cle.tf_framework.tf_lib
from sensor_msgs.msg import JointState

@nrp.MapSpikeSource("MF_pos_activity", nrp.map_neurons(range(0,50), lambda i: nrp.brain.MF_pos[i]), nrp.poisson, delay = 1.0)
@nrp.MapSpikeSource("MF_vel_activity", nrp.map_neurons(range(0,50), lambda i: nrp.brain.MF_vel[i]), nrp.poisson, delay = 1.0)
@nrp.MapRobotSubscriber("joints", Topic("/icub/joints", JointState))
@nrp.Robot2Neuron()
def sensory_activity (t, MF_pos_activity, MF_vel_activity, joints):
    joints = joints.value
    head_pos = joints.position[joints.name.index('neck_yaw')]
    head_vel = joints.velocity[joints.name.index('neck_yaw')]
        
    head_pos = ((head_pos + 0.8) / 1.6)
    head_vel = ((head_vel + 0.8 * 2 * 3.14) / (1.6 * 2 * 3.14))

    if head_pos > 1.0:
        head_pos = 1.0
    elif head_pos < 0.0:
        head_pos = 0.0
    if head_vel > 1.0:
        head_vel = 1.0
    elif head_vel < 0.0:
        head_vel = 0.0

    min_rate = 0.0
    max_rate = 600.0
    sigma = 0.02

    for i in range(50):
     	mean = float(i) / 50.0 + 0.01
       	gaussian = np.exp(-((head_pos - mean) * (head_pos - mean))/(2.0 * sigma * sigma))
       	MF_pos_activity[i].rate = min_rate + gaussian * (max_rate - min_rate)

    for i in range(50):
       	mean = float(i) / 50.0 + 0.01
       	gaussian = np.exp(-((head_vel - mean) * (head_vel - mean))/(2.0 * sigma * sigma))
       	MF_vel_activity[i].rate = min_rate + gaussian * (max_rate - min_rate)

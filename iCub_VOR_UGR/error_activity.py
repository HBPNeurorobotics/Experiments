import hbp_nrp_cle.tf_framework.tf_lib
from sensor_msgs.msg import JointState
@nrp.MapSpikeSource("IO_agonist_activity", nrp.map_neurons(range(0,100), lambda i: nrp.brain.IO_agonist[i]), nrp.poisson, delay = 20.0)
@nrp.MapSpikeSource("IO_antagonist_activity", nrp.map_neurons(range(0,100), lambda i: nrp.brain.IO_antagonist[i]), nrp.poisson, delay = 20.0)
@nrp.MapRobotSubscriber("joints", Topic("/icub/joints", JointState))
@nrp.MapCSVRecorder("error_rec", filename="retinal_slip_error.csv", headers=["t", "error"])
@nrp.Robot2Neuron()
def error_activity(t, IO_agonist_activity, IO_antagonist_activity, joints, error_rec):
    def compute_P_error(kp, head_position, eye_position):
        error = kp * (head_position + eye_position)
        return error
    def compute_D_error(kd, head_velocity, eye_velocity):
        error = kd * (head_velocity + eye_velocity)
        return error

    MAX_AMPLITUDE = 0.8

    joints = joints.value
    head_pos = joints.position[joints.name.index('neck_yaw')]
    eye_pos = joints.position[joints.name.index('eye_version')]
    kp=15.0
    position_error = compute_P_error(kp, head_pos, eye_pos)
    head_vel = joints.velocity[joints.name.index('neck_yaw')]
    eye_vel = joints.velocity[joints.name.index('eye_version')]
    kd=15.0
    velocity_error = compute_D_error(kd, head_vel, eye_vel)

    error=(position_error * 0.1 + (velocity_error/(2.0*3.141592)) * 0.9)/MAX_AMPLITUDE

    min_rate = 1.0
    max_rate = 25.0
    low_neuron_ID_threshold = abs(error) * 100.0
    up_neuron_ID_threshold = low_neuron_ID_threshold - 100.0
    rate = []
    for i in range (100):
        if(i < up_neuron_ID_threshold):
            rate.append(max_rate)
        elif(i<low_neuron_ID_threshold):
            aux_rate=max_rate - (max_rate-min_rate)*((i - up_neuron_ID_threshold)/(low_neuron_ID_threshold - up_neuron_ID_threshold))
            rate.append(aux_rate)
        else:
            rate.append(min_rate)
    if error>=0.0:
        IO_agonist_activity.rate=min_rate
        IO_antagonist_activity.rate=rate
    else:
        IO_antagonist_activity.rate=min_rate
        IO_agonist_activity.rate=rate

    error_rec.record_entry(t,error)

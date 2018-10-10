# Imported Python Transfer Function
#import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from std_msgs.msg import Float64
from hbp_nrp_excontrol.logs import clientLogger
import FORCE_ANN_NRP_4t as FAN
reload(FAN)
#import GlobalUtils as GU
import lpFilter
from tigrillo_2_plugin.msg import Motors, Sensors
import rospy

params = ['spec_rad', 'scale_w_fb', 'offset_w_fb', 'scale_noise_fb', 'offset_w_res', 'N_toKill', 'FORCE_dur',
               'delay', 'post_learn', 'alpha']

sol_denorm = [9.9952822022920138, 1.65, 0.25, 0.010313427017175601, 0.18, 64, 3.6, 3.6, 1.5, 0.12440507262210909] #from small cmaes search
#sol_denorm = [9.9952822022920138, 1.65, 0.25, 0.010313427017175601, 0.18, 64, 3., 3., 1.5, 0.12440507262210909] #short training


#exp_dir = GU.make_directory('ClAnnForce4t_NRPexperiment')

FORCE_ANN_params = dict((x, y) for x, y in zip(params, sol_denorm))
force_ann = FAN.force_ann_experiment(FORCE_ANN_params,res_size=800,target='resources/downsampled_cpg_model14052018_walking.txt', HLI=False) 
#downsampled_cpg_calibratedT_bounding_turning1direction 
#downsampled_cpg_calibratedT_bounding_turningviaamplitude 
#downsampled_cpg_calibratedT_bounding_turning 
#downsampled_cpg_calibratedT_9D 
#downsampled_cpg_calibratedT_7D 
#downsampled_cpg_realT_bounding_turning_8p9rs_stepsfrom0 
#/downsampled_cpg_walking.npy 
#/downsampled_cpg_walking_500-1400mHz.npy 
#downsampled_cpg_realT_9D downsampled_cpg_gait0_bounding_5s_smooth 
#downsampled_cpg_realT_bounding 
#downsampled_cpg_realT_bounding_800mHz downsampled_cpg_realT_bounding_turning downsampled_cpg_realT_bounding_turning_8p9rs downsampled_cpg_realT_bounding_turning_8p9rs_steps downsampled_cpg_realT_bounding_turning_8p9rs_stepsfrom0
#force_ann = FAN.force_ann_experiment(FORCE_ANN_params,res_size=200,target='/home/alexander/Dropbox/UGent/Code/Python/downsampled_cpg_model14052018_walking.npy', HLI=False) 

SaveTime = (10**FORCE_ANN_params['FORCE_dur'] + 10**FORCE_ANN_params['delay'] + 10**FORCE_ANN_params['post_learn'])*0.02 + 30


@nrp.MapVariable('SaveTime',initial_value = SaveTime) # time (s) of saving data
@nrp.MapVariable('lpfilter',initial_value = lpFilter.LPF())
#@nrp.MapVariable('exp_dir',initial_value = exp_dir)
@nrp.MapVariable('step',initial_value = -1) #count N steps (t keeps adding even if tf reset)
@nrp.MapVariable('Force_ann',initial_value = force_ann)

@nrp.MapVariable('output',initial_value = 0.0)

@nrp.MapVariable('pub',initial_value = rospy.Publisher("tigrillo_rob/uart_actuators", Motors, queue_size=1))
@nrp.MapRobotSubscriber("joint_states", Topic("tigrillo_rob/sim_sensors", Sensors)) #in this tf only for recording purposes

@nrp.Neuron2Robot()
def tf_ANN_FORCE(t, pub, output, Force_ann, step, joint_states, lpfilter, SaveTime):
    import time
    time0 = time.time()
    step.value = step.value+1 #increase step with 1
    msg = joint_states.value 
    if not isinstance(msg, type(None)):
        pos = [msg.FL, msg.FR, msg.BL, msg.BR]
        ### normalize joint readouts to feed to ANN
        offset = 5
        multiplier = 1./25
        inputNorm = np.array([pos]) + offset #output.value + offset
        inputNorm = inputNorm * multiplier
        ### low pass filter joint readouts
        inputNorm[0] = lpfilter.value.filterit(inputNorm[0])
        inputNorm[0][0] = (inputNorm[0][0]-0.2)*3  # = (inputNorm[0][1] - 0.4)*5
        inputNorm[0][1] = (inputNorm[0][1]-0.2)*3
        inputNorm[0][2] = inputNorm[0][2]*2
        inputNorm[0][3] = inputNorm[0][3]*2
        ### Update ANN with sensor input/its own output
        time1=time.time()
        output.value = Force_ann.value.step_quick(inputNorm,step.value)
        time2=time.time()
        ### send ANN/force output to motor
        pub.value.publish(run_time=t, FL=output.value[0][0], FR=output.value[0][1], BL=output.value[0][2], BR=output.value[0][3])
    return

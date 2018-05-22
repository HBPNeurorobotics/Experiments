# Imported Python Transfer Function
import numpy as np
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from std_msgs.msg import Float64
import sensor_msgs.msg
from hbp_nrp_excontrol.logs import clientLogger
import sys
import time
import force_algorithm3
import lpFilter
from tigrillo_2_plugin.msg import Motors, Sensors
import rospy
targetfile = 'resources/downsampled_cpg_model14052018_bounding.txt'
globals()['HLI'] = False
params = ['spec_rad', 'scale_w_fb', 'offset_w_fb', 'scale_noise_fb', 'offset_w_res', 'N_toKill', 'FORCE_dur', 'delay', 'post_learn', 'alpha']
sol_denorm = [9.5, 1.65, 0.25, 0.12, 0.18, 64, nrp.config.brain_root.FORCE_dur, nrp.config.brain_root.delay_dur, 0.0, -2.1] #3.65, 2.4,
FORCE_ANN_params = dict((x, y) for x, y in zip(params, sol_denorm))
N_readouts = len(nrp.config.brain_root.SNN.readout_neuron_populations)
res_size = nrp.config.brain_root.SNN.monitor_population.size
initial_weights = np.random.randn(res_size,N_readouts)*10**-1
alpha = 10.
delay = 10 ** FORCE_ANN_params['delay']
FORCE_dur = 10 ** FORCE_ANN_params['FORCE_dur']
post_learn = 10 ** FORCE_ANN_params['post_learn']
n_it = int(delay + FORCE_dur + post_learn + 1000)
force_snn = force_algorithm3.FORCE_Gradual(initial_weights, N_readouts, alpha, delay, FORCE_dur, post_learn)
header = ['time']
header.extend(['sensorPop' + str(i) for i in range(nrp.config.brain_root.SNN.sensor_monitor_population.size)])
header.extend(['resPop' + str(i) for i in range(res_size)])
@nrp.MapVariable('w_out', initial_value=initial_weights)
@nrp.MapVariable('step', initial_value=-1)  # count N steps (t keeps adding even if tf reset)
@nrp.MapVariable('Force_snn', initial_value=force_snn)
@nrp.MapVariable('target',initial_value=np.transpose(np.loadtxt(targetfile)))
@nrp.MapVariable('lpfilter', initial_value=lpFilter.LPF(order=3, fs=50.0, cutoff=7.0))
@nrp.MapVariable('readout_pops',initial_value =nrp.config.brain_root.SNN.readout_neuron_populations)
@nrp.MapVariable('monitor_pop',initial_value =nrp.config.brain_root.SNN.monitor_population)
@nrp.MapVariable('sensor_monitor_pop',initial_value =nrp.config.brain_root.SNN.sensor_monitor_population)
@nrp.MapVariable('pub',initial_value = rospy.Publisher("tigrillo_rob/uart_actuators", Motors, queue_size=1))
@nrp.MapRobotSubscriber("joint_states", Topic("tigrillo_rob/sim_sensors", Sensors)) #in this tf only for recording purposes
@nrp.Neuron2Robot()
def tf_SNN_FORCE(t, w_out, readout_pops, monitor_pop, sensor_monitor_pop, pub,  joint_states, Force_snn, target, step, lpfilter):
    import time
    step.value = step.value + 1  # increase step with 1
    ### fetch SNN data
    ## sensor population monitor
    sensor_monitor = [sensor_monitor_pop.value.get_data(clear=True).segments[-1].analogsignals[0][-1]][0]
    sensor_monitor = [float(x) for x in sensor_monitor]
    ## reservoir state (monitor neurons)
    reservoir_state = [monitor_pop.value.get_data(clear=True).segments[-1].analogsignals[0][-1]][0]
    reservoir_state = [float(x) for x in reservoir_state]
    readout = np.dot(reservoir_state,w_out.value)
    ### step force algorithm, new_w wil be None if learning is finished
    normal_target = target.value[step.value:step.value + 1, :]
    noisy_target = normal_target+np.random.randn(1,target.value.shape[1])*2
    output_force, new_w, error = Force_snn.value.step(np.mat(reservoir_state).T, np.array([readout]),  noisy_target , step.value)
    ### update readout weights
    if isinstance(new_w, np.ndarray):
        w_out.value = new_w
    ### send SNN/force output to motor
    pub.value.publish(run_time=t, FL=output_force[0][0], FR=output_force[0][1], BL=output_force[0][2], BR=output_force[0][3])
    return

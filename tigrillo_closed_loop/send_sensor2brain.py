# Imported Python Transfer Function
import sensor_msgs.msg
import hbp_nrp_cle.tf_framework.tf_lib
import hbp_nrp_cle.tf_framework as nrp
import lpFilter
import numpy as np
from tigrillo_2_plugin.msg import Motors, Sensors
import rospy
@nrp.MapVariable('step_index', global_key="step_index", initial_value=-1)
@nrp.MapVariable('lpfilter', initial_value=lpFilter.LPF())
@nrp.MapVariable('FORCE_dur', initial_value= 10 **nrp.config.brain_root.FORCE_dur)
@nrp.MapVariable('delay_dur', initial_value= 10 **nrp.config.brain_root.delay_dur)
@nrp.MapRobotSubscriber("joint_states", Topic("tigrillo_rob/sim_sensors", Sensors))
@nrp.MapSpikeSource("sensor_pop0", nrp.brain.sensor_population0, nrp.dc_source)
@nrp.MapSpikeSource("sensor_pop1", nrp.brain.sensor_population1, nrp.dc_source)
@nrp.MapSpikeSource("sensor_pop2", nrp.brain.sensor_population2, nrp.dc_source)
@nrp.MapSpikeSource("sensor_pop3", nrp.brain.sensor_population3, nrp.dc_source)
@nrp.MapSpikeSource("HLI_pop", nrp.brain.HLI_population, nrp.dc_source) 
@nrp.Robot2Neuron()
def send_sensor2brain(t, step_index, sensor_pop0, sensor_pop1, sensor_pop2, sensor_pop3, HLI_pop, joint_states, lpfilter, FORCE_dur, delay_dur):  # cpgVar, extraNode_neuron,
    dir(step_index)
    step_index.value = step_index.value + 1  # increase step with 1
    msg = joint_states.value 
    if not isinstance(msg, type(None)):
        pos = [msg.FL-10, msg.FR-10, msg.BL-10, msg.BR-10]
        ### normalize joint readouts to feed to ANN
        offset = 5
        multiplier = 1./25
        inputNorm = np.array([pos]) + offset #output.value + offset
        inputNorm = inputNorm * multiplier
        ### low pass filter joint readouts
        inputNorm[0] = lpfilter.value.filterit(inputNorm[0])
        inputNorm[0][0] = (inputNorm[0][0]-0.2)*3
        inputNorm[0][1] = (inputNorm[0][1]-0.2)*3
        inputNorm[0][2] = inputNorm[0][2]*2
        inputNorm[0][3] = inputNorm[0][3]*2
        ### add impulse noise during learning
        if t < (delay_dur.value+FORCE_dur.value)*0.02: 
            for idx, itm in enumerate(inputNorm[0]):
                new_itm = itm + np.random.randn()/100
                if np.random.rand() > 0.99:
                    new_itm = new_itm + np.random.rand() * 5 - 0.5
                    inputNorm[0][idx]=new_itm
        sensor_pop0.amplitude = inputNorm[0][0] * 0.04
        sensor_pop1.amplitude = inputNorm[0][1] * 0.04
        sensor_pop2.amplitude = inputNorm[0][2] * 0.04
        sensor_pop3.amplitude = inputNorm[0][3] * 0.04

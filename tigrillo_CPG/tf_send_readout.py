# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import numpy as np
from std_msgs.msg import Float64
from hbp_nrp_excontrol.logs import clientLogger

#@nrp.MapVariable("mv",initial_value=hbp_nrp_cle.tf_framework.config.brain_root.circuit.get_data('v').segments[0])
#@nrp.MapSpikeSink("l_shoulder_neuron", nrp.brain.readoutPop0, nrp.leaky_integrator_exp,weight=0.025)
#@nrp.MapSpikeSink("r_shoulder_neuron", nrp.brain.readoutPop1, nrp.leaky_integrator_exp,weight=0.025)
#@nrp.MapSpikeSink("l_hip_neuron", nrp.brain.readoutPop2, nrp.leaky_integrator_exp,weight=0.025)
#@nrp.MapSpikeSink("r_hip_neuron", nrp.brain.readoutPop3, nrp.leaky_integrator_exp,weight=0.025)


@nrp.MapVariable('readout_pops',initial_value =nrp.config.brain_root.readout_neuron_populations)

@nrp.MapRobotPublisher('l_shoulder', Topic('/robot/left_shoulder/cmd_pos', Float64))
@nrp.MapRobotPublisher('r_shoulder', Topic('/robot/right_shoulder/cmd_pos', Float64))
@nrp.MapRobotPublisher('l_hip', Topic('/robot/left_hip/cmd_pos', Float64))
@nrp.MapRobotPublisher('r_hip', Topic('/robot/right_hip/cmd_pos', Float64))

@nrp.MapCSVRecorder("recorder", filename="readout.csv", headers=['time', 'l_shoulder', 'r_shoulder', 'l_hip', 'r_hip'])

@nrp.Neuron2Robot()
def send_readout(t, readout_pops, l_shoulder, r_shoulder, l_hip, r_hip, recorder):
	
	readout = [float(pop.get_data(clear=True).segments[-1].analogsignalarrays[0][-1]) for pop in readout_pops.value]
	#clientLogger.info(len(readout_pops.value[0].get_data(clear=True).segments[-1].analogsignalarrays[0]))
	#clientLogger.info(float(readout_pops.value[0].get_data(clear=True).segments[-1].analogsignalarrays[0][-1]))

	#Normalize
	max_readout = 1600 #completely network dependent and empirical parameter !
	readout = [x/max_readout for x in readout]
	multiplier = [70.65, 71.31, 73.50, 72.55]
	offset = [41.92, 42.26, 79.65, 79.18]
	for i in range(len(readout)):
		readout[i] = readout[i]*multiplier[i]-offset[i]

	#clientLogger.info(readout)

	l_shoulder.send_message(readout[0]/57.3)# /57.3 to convert to radian
	r_shoulder.send_message(readout[1]/57.3)
	l_hip.send_message(readout[2]/57.3)
	r_hip.send_message(readout[3]/57.3)
    	
	#clientLogger.info("t (ms) : "+str(t))
	#clientLogger.info("device voltage: "+str(l_shoulder_neuron.voltage))
	#clientLogger.info(str(v))
	#clientLogger.info(mv.value.analogsignalarrays)

	recorder.record_entry(t,readout[0],readout[1],readout[2],readout[3])	


    	#return v

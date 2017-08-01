
@nrp.MapVariable('readout_pops',initial_value =nrp.config.brain_root.readout_neuron_populations)
@nrp.MapCSVRecorder("recorder", filename="readout.csv", headers=['time', 'l_shoulder', 'r_shoulder', 'l_hip', 'r_hip'])


@nrp.Neuron2Robot()

def csv_readout(t, readout_pops, recorder):

	# readout = membrane potential of readout neurons
	readout = [float(pop.get_data(clear=True).segments[-1].analogsignalarrays[0][-1]) for pop in readout_pops.value]

	# normalize readout
	max_readout = 1600 #completely network dependent and empirical parameter !
	readout = [x/max_readout for x in readout]
	multiplier = [70.65, 71.31, 73.50, 72.55]
	offset = [41.92, 42.26, 79.65, 79.18]
	for i in range(len(readout)):
		readout[i] = readout[i]*multiplier[i]-offset[i]
	
	# record entry
	recorder.record_entry(t,readout[0],readout[1],readout[2],readout[3])	


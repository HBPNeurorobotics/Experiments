    # Imported Python Transfer Function
    #
    #This TF saves the neuron spikes on a file only
    from std_msgs.msg import Empty
    @nrp.MapVariable("brain_data", initial_value=None)
    @nrp.MapSpikeSink("all_neurons", nrp.brain.circuit[slice(0, 1280, 1)], nrp.spike_recorder)
    @nrp.Neuron2Robot(Topic('/dummy', Empty))
    def all_neurons_spike_monitor(t, brain_data, all_neurons):
        pass
        #ns = brain_data.value
        #if ns is None:
        #    ns = [0]*1280
        #for q in all_neurons.times.tolist():
        #    ns[int(q[0])-nrp.config.brain_root.circuit[0]] = q[1]
        #nss = map(str, ns)
        #brain_data.value = ns
    #

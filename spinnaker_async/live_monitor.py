# Imported Python Transfer Function
@nrp.MapSpikeSink("rec", nrp.brain.right, nrp.spike_recorder)
@nrp.Neuron2Robot(triggers="rec")
def live_monitor(t, rec):
    clientLogger.info("Spikes recorded: {}".format(rec.times))
##


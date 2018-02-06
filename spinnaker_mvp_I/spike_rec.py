import hbp_nrp_cle.tf_framework as nrp

@nrp.MapSpikeSink("population", nrp.brain.right, nrp.spike_recorder)
@nrp.Neuron2Robot(triggers=["population"])
def spike_rec(t, population):
    clientLogger.info("Right output spiked: {}".format(population.times))
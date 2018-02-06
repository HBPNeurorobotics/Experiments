import hbp_nrp_cle.tf_framework as nrp

@nrp.MapSpikeSink("population", nrp.brain.right, nrp.leaky_integrator_exp, weight=2.0)
@nrp.Neuron2Robot(triggers=["population"])
def leaky_integrator(t, population):
    clientLogger.info("Leaky integrate: {}".format(population.voltage))

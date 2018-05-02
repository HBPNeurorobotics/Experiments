import hbp_nrp_cle.tf_framework as nrp

@nrp.MapSpikeSource("population", nrp.brain.left, nrp.injector, weight=2.0, n=3)
def inject(t, population):
    clientLogger.info("Injecting spikes")
    population.inject_spikes()
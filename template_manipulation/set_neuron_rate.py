# Imported Python Transfer Function
#
@nrp.MapSpikeSource("input_neuron", nrp.brain.neurons[0], nrp.poisson)
@nrp.Robot2Neuron()
def set_neuron_rate(t, input_neuron):
    input_neuron.rate = 10
#


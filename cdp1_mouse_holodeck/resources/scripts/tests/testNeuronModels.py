import sys
sys.path.append('../code')
import pyNN.nest as sim  # can of course replace `neuron` with `nest`, `brian`, etc.
import matplotlib.pyplot as plt
from CellsParameters import CellsParameters
import numpy as np

def main():
    """ Test neurons behaviors """
    
    sim.setup(timestep=0.01)
    p_in = sim.Population(1, sim.SpikeSourceArray(spike_times=[100]), label="input")
    p_out = sim.Population(1, CellsParameters.motoneurons("if_curr_alpha"), label="test neurons")

    syn = sim.StaticSynapse(weight=-1,delay=1)
    conn = sim.AllToAllConnector()
    # connections = sim.Projection(p_in, p_out, conn, syn, receptor_type='excitatory')
    connections = sim.Projection(p_in, p_out, conn, syn, receptor_type='inhibitory')

    p_in.record('spikes')
    p_out.record('spikes')                    # record spikes from all neurons
    p_out.record(['v'])  # record other variables from first two neurons

    sim.run(500.0)

    spikes_in = p_in.get_data()
    data_out = p_out.get_data()


    fig_settings = {
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
        'font.size': 8
    }
    plt.rcParams.update(fig_settings)
    plt.figure(1, figsize=(16,4))
    n_panels = sum(a.shape[1] for a in data_out.segments[0].analogsignalarrays) + 1
    ax = plt.subplot(n_panels, 1, 1)
    plot_spiketrains(spikes_in.segments[0])
    plt.subplot(n_panels, 1, 1)
    plot_spiketrains(data_out.segments[0])
    panel = 2
    for array in data_out.segments[0].analogsignalarrays:
        for i in range(array.shape[1]):
            plt.subplot(n_panels, 1, panel,sharex=ax)
            plot_signal(array, i, colour='bg'[panel % 2])
            panel += 1
    plt.xlabel("time (%s)" % array.times.units._dimensionality.string)
    plt.setp(plt.gca().get_xticklabels(), visible=True)

    plt.show()


def plot_spiketrains(segment):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
        plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)


def plot_signal(signal, index, colour='b'):
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], colour, label=label)
    print "EPSP is: %f" % (np.max(signal[:, index])-np.min(signal[:, index]))
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()

if __name__ == "__main__":
    main()

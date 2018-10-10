
import numpy as np
import matplotlib.pyplot as plt
import nest
import pickle
#import SpiNNNetwork4
import datetime
import os
import neo

def get_population_spikes(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spikearrays containing spiketimes per population
    """
    spikes_arraylist = []
    for idx, pop in enumerate(pop_list):
        spiketimes = nest.GetStatus(pop.recorder._spike_detector.device, 'events')[0]['times']
        spikes_arraylist.append(spiketimes)
    return spikes_arraylist

def get_population_spikes_in_2D_array(pop_list):
    """ for NEST
    retruns a 2d array of spiketimes for each population in pop_list"""
    spiketimes_2D = []
    #Tracer()()
    for idx, pop in enumerate(pop_list):
        spiketimes = nest.GetStatus(pop.recorder._spike_detector.device, 'events')[0]['times']
        IDs=nest.GetStatus(pop.recorder._spike_detector.device,'events')[0]['senders']
        if IDs.shape[0] == 0:
            spiketimes_2D.append([])
        else:
            minID = min(IDs)
            maxID = max(IDs)
            N = maxID-minID
            pop_neuron_spiketimes = []
            for i in range(minID,maxID+1):
                indices = np.where(IDs==i)[0]
                pop_neuron_spiketimes.append(spiketimes[indices])
            spiketimes_2D.append(pop_neuron_spiketimes)
    return np.array(spiketimes_2D)

def get_population_spikes_spiNNaker(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spikearrays containing spiketimes per population
    """
    spikes_arraylist = []
    for idx, pop in enumerate(pop_list):
        spiketimes = pop.getSpikes()
        spikes_arraylist.append(spiketimes)
    return spikes_arraylist

def get_spiketrains(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return:    list with spiketrains
    """
    spiketrains_list = []
    for idx, pop in enumerate(pop_list):
        spiketrains = pop.get_data('spikes').segments[0].spiketrains
        spiketrains_list.append(spiketrains)
    return spiketrains_list


def retrieve_voltage_data(pop_list):
    """
    :param pop_list: list with pyNN populations
    :return: v_list: list with voltage values
    """
    if not isinstance(pop_list,list):
        raise Exception('input parameter should be a list')
    v_list =[]
    for pop in pop_list:
        try:#nest 2.10
                v= pop.get_data('v').segments[0].analogsignalarrays[0]
        except:#nest 2.12
                v= pop.get_data('v').segments[0].analogsignals[0]
                
        v2 = []
        for i in range(v.shape[1]):  # get data out of that really annoying format
            signal = v[:, i]
            signal = np.array([float(s) for s in signal])
            v2.append(signal)
        v_list.append(np.array(v2))
        times = [float(t) for t in v.times]
    return v_list, times

def retrieve_voltage_data_spiNNaker(pop_list,N=None):
    """
    :param pop_list: list with pyNN populations
    :param N:        max number of populations to get voltages from
    :return: v_list: list with voltage values
    """
    if not isinstance(pop_list,list):
        raise Exception('input parameter should be a list')

    if N==None or N>len(pop_list):
        N=len(pop_list)


    v_list =[]
    for pop in pop_list[:N]:
        v= pop.get_v()
        v_list.append(v)
    return v_list

def get_population_activities(spikes, timebin, start, stop, pop_size):
    """
    calculate Population Activity (average number of spikes per timebin)
    :param spikes: [ np.array(spiketimes), ... ]
    :param timebin: ms, bin width to calculate population activity
    :param start:  ms, start time from where to calculate population activity
    :param stop: ms, stop time until where to calculate population activity
    :param pop_size: population size
    :return: population_activities, [[fl, fl, ...], [], ...] list of lists of floats
    """
    population_activities = []

    for spiketimes in spikes:
        population_activity = []
        for t in np.arange(start, stop, timebin):
            if t+timebin>stop: # dont calculate PA if bin not in [start,stop]
                continue
            Nspikes_pop = len(np.where(np.logical_and(spiketimes >= t, spiketimes < t + timebin))[0])
            # if Nspikes_pop>0:
                # Tracer()()
            PA = (Nspikes_pop / float(pop_size)) / (0.001 * timebin)  # population activity in Hz
            population_activity.append(PA)
        population_activities.append(population_activity)
    return population_activities

def plot_simulation(spiketrains_in,spiketrains_hidden_exc,spiketrains_hidden_inh,population_activities,input_population_activities,PA_timebin,res_size,input_size,sim_duration,hiddenP_size,hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations_exc=None):
    """

    :param spiketrains_in:
    :param spiketrains_hidden_exc:
    :param spiketrains_hidden_inh:
    :param population_activities: population_activities, [[fl, fl, ...], [], ...] list of lists of floats
    :param input_population_activities:
    :param PA_timebin:
    :param res_size:
    :param input_size:
    :param sim_duration:
    :param hiddenP_size:
    :param hiddenPexc_size:
    :param plot_voltage_traces:
    :param v_hidden_populations_exc:
    :return:
    """
    N_hidden_populations = res_size
    # change figure size
    plt.rcParams["figure.figsize"][0] = 11.0
    plt.rcParams["figure.figsize"][1] = 16.0

    fig0 = plt.figure()

    # subplot 1 - plot input spikes
    ax1 = fig0.add_subplot(411)
    for idx in range(input_size):
        spiketrains_input = spiketrains_in[idx]
        for st in spiketrains_input:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenPexc_size + 10)
            ax1.plot(st, y, '|', color='blue', mew=2, markersize=1.5)
    ax1.set_xlim(0, sim_duration)
    ax1.set_ylabel('Neuron ID')
    ax1.set_title('Input Spikes')

    # subplot 2 - plot hiddenP spikes
    if plot_voltage_traces:  # if voltage traces to be plotted use only one row, else two
        ax2 = fig0.add_subplot(412, sharex=ax1)
    elif not (plot_voltage_traces):
        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=2, sharex=ax1)
    for idx in range(N_hidden_populations):
        spiketrains_hiddenPexc = spiketrains_hidden_exc[idx]
        spiketrains_hiddenPinh = spiketrains_hidden_inh[idx]
        for st in spiketrains_hiddenPexc:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenP_size + 10)
            ax2.plot(st, y, '|', color='black', mew=2, markersize=1.5)
        for st in spiketrains_hiddenPinh:
            y = np.ones((st.size)) * st.annotations['source_index'] + idx * (hiddenP_size + 10) + hiddenPexc_size
            ax2.plot(st, y, '|', color='red', mew=2, markersize=1.5)
    ax2.set_xlim(0, sim_duration)
    ax2.set_ylabel('Neuron ID')
    ax2.set_title('Hidden Population Spikes')
    ax2.legend()

    # subplot 3 (optional) - plot voltage of first X excitatory reservoir neurons
    if plot_voltage_traces:
        Nvoltage_traces = 50
        ax3 = fig0.add_subplot(413, sharex=ax1)
        for idx in range(1):#N_hidden_populations
            v = v_hidden_populations_exc[idx]
            for x in range(Nvoltage_traces):
                signal = v[:, x]
                signal = np.array([float(s) for s in signal])
                ax3.plot(v.times, signal + 30 * x + idx * 30 * Nvoltage_traces, color='black')

        ax3.set_ylabel('membrane Voltage (mV)')
        ax3.set_title('membrane potential of x Hidden Population neurons')
        ax3.legend()

    # subplot 4 - plot population activity
    ax4 = fig0.add_subplot(414, sharex=ax1)
    for idx in range(N_hidden_populations):
        population_activity = np.array(population_activities[idx])
        if idx == 0:  # add label only once
            ax4.plot(np.linspace(0, sim_duration, len(population_activity)), population_activity + idx * 1000, color='black',
                     label='Population/Monitor Neuron Activity')
        else:
            ax4.plot(np.linspace(0, sim_duration, len(population_activity)), population_activity + idx * 1000, color='black')
    summed_input_activities = [sum(x) for x in zip(*input_population_activities)]
    for idx in range(input_size):
        if idx ==0: # add label only once
            ax4.plot(np.linspace(0, sim_duration, len(input_population_activities[idx])), np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue',
             label='Input Population Activity')
        else:
            ax4.plot(np.linspace(0, sim_duration, len(input_population_activities[idx])),np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue')
    ax4.set_xlabel("time (ms)")
    ax4.set_ylabel("Activity (Hz)")
    ax4.set_ylim(26800,28500)
    ax4.legend()

    # plot
    plt.tight_layout()
    plt.show()

    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0
    return

def plot_simulation_spiNNaker(spikes_in,spikes_hidden_exc,spikes_hidden_inh,population_activities,input_population_activities,PA_timebin,res_size,input_size,sim_duration,hiddenP_size,hiddenPexc_size,plot_voltage_traces=False,v_hidden_populations_exc=None,readout_spikes=None,readout_activities=None):
    """ plotting voltage traces still needs some adaptations"""
    N_hidden_populations = res_size
    # change figure size
    plt.rcParams["figure.figsize"][0] = 11.0
    plt.rcParams["figure.figsize"][1] = 16.0

    fig0 = plt.figure()

    # subplot 1 - plot input spikes
    ax1 = fig0.add_subplot(411)
    for idx in range(input_size):
        spikes = spikes_in[idx]
        y = spikes[:,0] + idx * (hiddenPexc_size + 10)
        ax1.plot(spikes[:,1], y, '|', color='blue', mew=2, markersize=1.5)
    ax1.set_xlim(0, sim_duration)
    ax1.set_ylabel('Neuron ID')
    ax1.set_title('Input Spikes')

    # subplot 2 - plot hiddenP spikes
    if plot_voltage_traces:  # if voltage traces to be plotted use only one row, else two
        ax2 = fig0.add_subplot(412, sharex=ax1)
    elif not (plot_voltage_traces):
        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=2, sharex=ax1)
    for idx in range(N_hidden_populations):
        spikes_exc = spikes_hidden_exc[idx]
        spikes_inh = spikes_hidden_inh[idx]

        y_exc = spikes_exc[:, 0] + idx * (hiddenP_size + 10)
        ax2.plot(spikes_exc[:,1], y_exc, '|', color='black', mew=2, markersize=1.5)

        if spikes_inh.shape[0]>0:
            y_inh = spikes_inh[:, 0] + idx * (hiddenP_size + 10) + hiddenPexc_size
            ax2.plot(spikes_inh[:,1], y_inh, '|', color='red', mew=2, markersize=1.5)

    if readout_spikes != None:
        for idx,itm in enumerate(readout_spikes):
            if itm.shape[0]>0:
                y_readout = itm[:, 0]-(hiddenPexc_size+10)*(idx+1)
                ax2.plot(itm[:, 1],y_readout,'|', color='orange',mew=2,markersize=2)

    ax2.set_xlim(0, sim_duration)
    ax2.set_ylabel('Neuron ID')
    ax2.set_title('Hidden Population Spikes')
    ax2.legend()

    # subplot 3 (optional) - plot voltage of first X excitatory neurons per population
    if plot_voltage_traces:
        Nvoltage_traces = 3
        ax3 = fig0.add_subplot(413, sharex=ax1)
        for idx in range(len(v_hidden_populations_exc)):
            v_hiddenPexc = v_hidden_populations_exc[idx]
            for neuronID in range(Nvoltage_traces):
                voltages = v_hiddenPexc[np.where(v_hiddenPexc[:,0]==neuronID)]
                x = voltages[:,1]
                y = voltages[:,2]
                ax3.plot(x, y + 30 * neuronID + idx * 30 * 2 * Nvoltage_traces, color='black')

        ax3.set_ylabel('membrane Voltage (mV)')
        ax3.set_title('membrane potential of x Hidden Population neurons')
        ax3.legend()

    if population_activities != None:
        # subplot 4 - plot population activity
        ax4 = fig0.add_subplot(414, sharex=ax1)
        for idx in range(N_hidden_populations):
            population_activity = np.array(population_activities[idx])
            if idx == 0:  # add label only once
                ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black',
                         label='Population Activity')
            else:
                ax4.plot(np.arange(0, sim_duration, PA_timebin), population_activity + idx * 1000, color='black')
        summed_input_activities = [sum(x) for x in zip(*input_population_activities)]
        for idx in range(input_size):
            if idx ==0: # add label only once
                ax4.plot(np.arange(0, sim_duration, PA_timebin), np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue',
                 label='Input Population Activity')
            else:
                ax4.plot(np.arange(0, sim_duration, PA_timebin),np.array(input_population_activities[idx]) + (res_size+idx)*1000, color='blue')
        if readout_activities!=None:
            for activities in readout_activities:
                ax4.plot(np.arange(0, sim_duration, PA_timebin),np.array(activities)-1000, color='orange')
        ax4.set_xlabel("time (ms)")
        ax4.set_ylabel("Activity (Hz)")
        # ax4.set_ylim(0,45000)
        ax4.legend()

    # plot
    plt.tight_layout()
    plt.show()

    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0
    return

def plot_monitor_voltages(v, times=None, tick_width=1000):
    """
    :param times : 1D array, times in ms
    :param v: sequence shape (Npops, Nrecordings)
    :return:
    """
    plt.figure()
    if times != None:
        for idx, itm in enumerate(v):
            plt.plot(times,itm + idx * tick_width)
            plt.xlabel("time (ms)")
    else:
        for idx, itm in enumerate(v):
            plt.plot(itm + idx * tick_width)
            plt.xlabel("time (?ms?)")
    plt.ylabel('membrane potential (mV)')
    plt.title("monitor_voltages")
    plt.show()
    return

def plot_spiketrain(spiketrains,pop_size=100):

    plt.figure()
    for idx, pop_trains in enumerate(spiketrains):
        for train in pop_trains:
            y = np.ones((train.size)) * train.annotations['source_index'] + idx * (pop_size + 10)
            plt.plot(train, y, '|', color='black', mew=2, markersize=1.5)
    # ax1.set_xlim(0, sim_duration)
    plt.ylabel('Neuron ID')
    plt.title('Spikes')
    plt.show()
    return

def compare_states(ANN_states,SNN_states,res_size,n_it):
    """
    :param ANN_states: states of rate-coded ANN
    :param SNN_states: states of SNN
    :param res_size: number of nodes in reservoir
    :param n_it: number of iterations
    :return:
    """
    """
    # Calculate mrse
    rse = 0
    for i in range(res_size):
        rse += np.sum(np.sqrt(np.square(ANN_states[:, i][20:] - np.array(SNN_states[i][21:]) / 500.0)))
    mrse = rse / (res_size * (n_it - 20))
    """

    # set plot window size
    plt.rcParams["figure.figsize"][0] = 3.0
    plt.rcParams["figure.figsize"][1] = 12.0
    fig1 = plt.figure()
    # plot SNN_states
    [plt.plot([y / 500 + i * 1.0 for y in SNN_states[i]], color='black') for i in range(res_size)]
    plt.plot(0, 0, color='black', label="Population Activity")  # for label
    # plot ANN_states
    if ANN_states != None:
        [plt.plot(range(n_it-1), ANN_states[:, x] + x * 1.0, color='green') for x in range(res_size)]
        plt.plot(0, 0, color='green', label="Rate-based neuron state")  # for label
    plt.legend()
    plt.xlabel("timesteps")
    plt.ylabel("PopulationID")
    # plt.title("MRSE = " + str(mrse) + " (discarded first 20 timesteps)")
    plt.show()
    # reset default plot window size
    plt.rcParams["figure.figsize"][0] = 8.0
    plt.rcParams["figure.figsize"][1] = 6.0

def saveSNN2File(dic):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    if not os.path.isdir(os.getcwd()+'/SNNFiles'):
        os.mkdir(os.getcwd()+'/SNNFiles')

    with open(os.getcwd()+'/SNNFiles/'+timestamp + '.SNN', 'w') as f:
            pickle.dump(dic, f)
    return

def loadSNNFromFile(filename):
    with open(filename, 'r') as f:
        NetworkData = pickle.load(f)
    return NetworkData

def load_network_from_file_spiNNaker(filename):
    """

    :param filename: string, location of .pkl file
    :return: Network
    """
    with open(filename, 'r') as f:
        NetworkData = pickle.load(f)

    Network = SpiNNNetwork4.Network(Pconnect=NetworkData['Pconnect'],
                                    num_hidden_populations=NetworkData['num_hidden_populations'],
                                    res_weights=NetworkData['res_weights']
                                    , feedback_weights=NetworkData['feedback_weights'],
                                    N_readouts=NetworkData['N_readouts'],
                                    readout_weights=NetworkData['readout_weights'],
                                    p_connect=NetworkData['p_connect'], Pconnect_fb=NetworkData['Pconnect_fb'])
    return Network


def getCoordinates(ID, xD, yD):
    """
    place population in grid based on id
    :param ID: id of population, long int
    :param xD: x dimensionality of grid, long int
    :param yD: y dimensionality of grid, long int
    :return: cartesian coordinates
    """
    if not (isinstance(ID, (int, long)) & isinstance(xD, (int, long)) & isinstance(yD, (int, long))):
        raise Exception('population ID, xDimension and yDimension must be integer types')
    zD = xD * yD

    z = ID / zD
    y = (ID - z * zD) / xD
    x = ID - z * zD - y * xD
    return x, y, z


def getProb(ID0, ID1, xD, yD, C=0.3, lamb=1.0):
    """
    get distance-based connection probability for pair (ID0,ID1)
    :param ID0: id of population 0
    :param ID1: id of population 1
    :param xD: x dimensionality of grid
    :param yD: y dimensionality of grid
    :param C: parameter to weight connectivity based on connection type (not yet implemented, from maass 2002)
    :param lamb: parameter to in/decrease overall connectivity
    :return: Probability of connection between any two neurons of populations ID0 and ID1
    """
    if ID0 == ID1:
        prob = 0.
    else:
        x0, y0, z0 = getCoordinates(ID0, xD, yD)
        x1, y1, z1 = getCoordinates(ID1, xD, yD)
        d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)  # eucl distance
        prob = C * np.power(np.e, -np.square(d / lamb))
    return prob

def createConnectivityMatrix(N,lamb=1.0):
    """
    create distance based connectivity matrix for N populations
    currently defaults to stacking populations in 3by3 layers

    :param N: number of populations
    :return: connectivity matrix (to, from)
    """
    p_connect = np.empty((N, N))
    for fr in range(N):
        for to in range(N):
            p_connect[fr, to] = getProb(to, fr, xD=3, yD=3, lamb=lamb)  # (to,from)
    return p_connect


def get_rand_mat(dim, spec_rad, negative_weights=True, seed=None):
    "Return a square random matrix of dimension @dim given a spectral radius @spec_rad"
    
    if seed:
        rng = np.random.RandomState(seed=seed)
        mat = rng.randn(dim, dim)
    else:
        mat = np.random.randn(dim, dim)
    if not (negative_weights):
        mat = abs(mat)
    w, v = np.linalg.eig(mat)
    mat = np.divide(mat, (np.amax(np.absolute(w)) / spec_rad))

    return mat

def InitReservoirNet(n_in=4, n_res=100, spec_rad=1.15, scale_in=5.0,
                 negative_weights=True):
    """
        creates initial weights and connection probabilities
    """

    w_res = get_rand_mat(n_res, spec_rad,negative_weights=negative_weights) # (to,from)
    # close autoconnections
    np.fill_diagonal(w_res, 0.0)

    w_in = np.random.randn(n_res, n_in) * scale_in

    if not(negative_weights):
        w_in = abs(w_in)

    # close half of feedback connections
    toKill = np.random.choice(range(n_res), int(n_res/ 2), replace=False)
    w_in[toKill] = 0.0

    p_connect_in = np.ones((n_res))*0.1 # fixed Pconnect for feedback connections
    p_connect_in = p_connect_in.reshape(-1,1)

    p_connect_res = createConnectivityMatrix(n_res) # (to,from)

    return w_in, w_res, p_connect_in, p_connect_res

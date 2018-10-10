"""

"""
import pyNN.nest as sim
#import matplotlib.pyplot as plt
#from matplotlib import cm
import numpy as np
import time
#import RC_Utils
#import Population_Utils as PU

class SNN_column():

    def __init__(self, SNN_dict):
        #all params
        networkStructure = SNN_dict['networkStructure']
        
        n_in = networkStructure['n_in']['n_sensor']
        n_res = networkStructure['n_res']
        n_readout = networkStructure['n_out']
        n_hli = networkStructure['n_in']['n_hli'] 

        p_connect_sensor = SNN_dict['p_connect']['p_connect_in']['p_connect_sensor']
        p_connect_input_hli = SNN_dict['p_connect']['p_connect_in']['p_connect_hli']
        p_connect_fb = SNN_dict['p_connect']['p_connect_in']['p_connect_fb']
        p_connect_inter = SNN_dict['p_connect']['p_connect_res']
        p_connect_intra_EE = SNN_dict['p_connect']['p_connect_intra']['EE']
        p_connect_intra_EI = SNN_dict['p_connect']['p_connect_intra']['EI']
        p_connect_intra_IE = SNN_dict['p_connect']['p_connect_intra']['IE']

        w_sensor = SNN_dict['weights']['w_in']['w_sensor']
        w_hli = SNN_dict['weights']['w_in']['w_hli']
        w_in = np.hstack((w_sensor,w_hli))
        w_gb = SNN_dict['weights']['w_in']['w_fb']
        w_res = SNN_dict['weights']['w_res']
        w_out = SNN_dict['weights']['w_out']
        w_intra_EE, w_intra_EI, w_intra_IE = SNN_dict['weights']['w_intra']['EE'], SNN_dict['weights']['w_intra']['EI'], SNN_dict['weights']['w_intra']['IE']

        noise_SD = SNN_dict['noise_SD']
        max_delay = SNN_dict['max_delay']
        interPop_delay = SNN_dict['delays']

        Nexc = SNN_dict['NexcNinh']['Nexc']
        Ninh = SNN_dict['NexcNinh']['Ninh']

        # NetworkData = loadSNNFromFile(filename)

        neuron_parameters = SNN_dict['neuron_parameters']['regular']

        readout_parameters = SNN_dict['neuron_parameters']['monitor']

        # initialize
        self.timestep = 1.0
        sim.setup(timestep=self.timestep, min_delay=1.0, max_delay=max_delay, threads=1, rng_seeds=[1234])

        # create connectors
        self.create_connectors(p_connect_intra_EE, p_connect_intra_EI, p_connect_intra_IE)
        # create pops
        self.create_populations(n_in, n_res, n_readout, n_hli, Nexc, Ninh, neuron_parameters, readout_parameters)
        # create proj
        self.create_projections(n_in, n_res, n_readout, n_hli, w_in, w_res, w_out, Nexc, p_connect_sensor, p_connect_inter, w_intra_EE, w_intra_EI, w_intra_IE, interPop_delay)
        # inject noises
        if noise_SD != 0:
                self.inject_noise(noise_SD)


    def create_connectors(self, p_connect_intra_EE, p_connect_intra_EI, p_connect_intra_IE):

        # =================================================================================================================
        # create connectors
        # =================================================================================================================
        self.connector_intra_EE = sim.FixedProbabilityConnector(p_connect=p_connect_intra_EE)
        self.connector_intra_EI = sim.FixedProbabilityConnector(p_connect=p_connect_intra_EI)
        self.connector_intra_IE = sim.FixedProbabilityConnector(p_connect=p_connect_intra_IE)

    def create_populations(self, n_in, n_res, n_readout, n_hli, Nexc, Ninh, neuron_parameters, readout_parameters):

        # =================================================================================================================
        # create populations
        # =================================================================================================================
        self.sensor_populations = []
        for idx in range(n_in+n_hli):
            self.sensor_populations.append(sim.Population(Nexc, sim.IF_curr_exp, neuron_parameters))

        self.sensor_monitor_population = sim.Population(n_in+n_hli, sim.IF_curr_exp, readout_parameters)

        self.hidden_populations = [[] for x in range(n_res)]
        for idx in range(n_res):
            hiddenPexc = sim.Population(Nexc, sim.IF_curr_exp, neuron_parameters, label='hiddenP' + str(idx))
            hiddenPinh = sim.Population(Ninh, sim.IF_curr_exp, neuron_parameters)
            self.hidden_populations[idx].append(hiddenPexc)
            self.hidden_populations[idx].append(hiddenPinh)

        self.monitor_population = sim.Population(n_res, sim.IF_curr_exp, readout_parameters)

        self.readout_neuron_populations = []
        for i in range(n_readout):
            pop = sim.Population(1, sim.IF_curr_exp, readout_parameters)
            self.readout_neuron_populations.append(pop)

        #self.dummy_hli_pop = sim.Population(1, sim.IF_curr_exp, readout_parameters)#due to nrp transfer function restricted python decorator labyrinth

    def create_projections(self, n_in, n_res, n_readout, n_hli, w_in, w_res, w_out, Nexc, p_connect_sensor, p_connect_inter, w_intra_EE, w_intra_EI, w_intra_IE, interPop_delay):

        # =====================================================================================================
        # create projections
        # =====================================================================================================
        # projections_sensor_monitor
        for idx in range(n_in+n_hli):
            conn_list = [(x, idx) for x in range(Nexc)]
            connector = sim.FromListConnector(conn_list)
            projection = sim.Projection(self.sensor_populations[idx], self.sensor_monitor_population, connector,
                                        sim.StaticSynapse(weight=1.0))

        # projections_sensor_hiddenP
        for idx0, pop in enumerate(self.hidden_populations):
            if sum(w_in[idx0]) != 0:
                for idx1 in range(n_in+n_hli):
                    projection_inp_hiddenP = sim.Projection(self.sensor_populations[idx1], pop[0],
                                                            sim.FixedProbabilityConnector(p_connect=p_connect_sensor),
                                                            sim.StaticSynapse(weight=w_in[idx0, idx1]))

        # projections_hiddenP_monitor
        for idx in range(n_res):
            connector = sim.FromListConnector(conn_list=[(x, idx) for x in range(Nexc)])
            projection = sim.Projection(self.hidden_populations[idx][0], self.monitor_population, connector,
                                        sim.StaticSynapse(weight=1.0))

        # projections_hiddenP_readout
        projections_hiddenP_readout = []
        connector = sim.AllToAllConnector()  # , delays=np.random.randint(6,72), rng=rng
        for idx0 in range(n_readout):
            readout_pop = self.readout_neuron_populations[idx0]
            projections = []
            for idx1 in range(n_res):
                pop0exc = self.hidden_populations[idx1][0]
                projection = sim.Projection(pop0exc, readout_pop, connector,
                                            sim.StaticSynapse(weight=w_out[idx0, idx1]))
                projections.append(projection)
            projections_hiddenP_readout.append(projections)

        # projections_hiddenP_hiddenP
        rng = sim.NumpyRNG(seed=2007200)
        projections_hiddenP_hiddenP = []
        for idx0 in range(n_res):
            pop0exc = self.hidden_populations[idx0][0]
            for idx1 in range(n_res):
                pop1exc = self.hidden_populations[idx1][0]
                projection = sim.Projection(pop0exc, pop1exc,
                                            sim.FixedProbabilityConnector(p_connect=p_connect_inter[idx1, idx0],
                                                                          rng=rng),
                                            sim.StaticSynapse(weight=w_res[idx1, idx0],delay=interPop_delay[idx1, idx0]))
                projections_hiddenP_hiddenP.append(projection)

        # create intrapopulation projections
        for idx in range(n_res):
            hiddenPexc = self.hidden_populations[idx][0]
            hiddenPinh = self.hidden_populations[idx][1]
            projection_intra_EI = sim.Projection(hiddenPexc, hiddenPinh, self.connector_intra_EI,
                                                              sim.StaticSynapse(weight=w_intra_EI))
            projection_intra_EE = sim.Projection(hiddenPexc, hiddenPexc, self.connector_intra_EE,
                                                              sim.StaticSynapse(weight=w_intra_EE))
            projection_intra_IE = sim.Projection(hiddenPinh, hiddenPexc, self.connector_intra_IE,
                                                              sim.StaticSynapse(weight=w_intra_IE))

    def inject_noise(self, noise_SD):

        # create and inject noise
        white_noise_input = sim.NoisyCurrentSource(mean=0.0, stdev=noise_SD, start=0.0, dt=self.timestep)
        for pop in self.sensor_populations:
            pop.inject(white_noise_input)

        white_noise_res = sim.NoisyCurrentSource(mean=0.0, stdev=noise_SD, start=0.0, dt=self.timestep)
        for pop in self.hidden_populations:
            pop[0].inject(white_noise_res)

        #white_noise_input_monitor = sim.NoisyCurrentSource(mean=0.0, stdev=0.0, start=0.0, stop=55000, dt=timestep)
        #monitor_population.inject(white_noise_input_monitor)

    def inject_impulse(self, start, stop, amplitude):
        dc = sim.DCSource(start=start, stop=stop, amplitude=amplitude)
        for pop in self.sensor_populations:
            pop.inject(dc)
        return

    def inject_sine(self, start, stop, amplitude, offset, frequency):
        sine = sim.ACSource(start=start, stop=stop, amplitude=amplitude, offset=offset, frequency=frequency)
        for pop in self.sensor_populations:
            pop.inject(sine)
        return


    def set_recordings(self):

        # set recordings
        # hidden_populations[0][0].record(['v'])
        for pop in self.readout_neuron_populations:
            pop.record(['v'], sampling_interval=5.0)  # sample every 5 ms

        self.sensor_monitor_population.record(['v'], sampling_interval=5.0)  # sample every 5 ms
        self.monitor_population.record(['v'], sampling_interval=5.0)
        #self.hidden_populations[0][0].record(['spikes'])
        #self.hidden_populations[0][1].record(['spikes'])

    def run_sim(self, duration):

        sim.run(duration)

    def get_data(self):
        pass
    def save_SNN(self):
        pass

class SNN_column_negW(SNN_column):
        """
        neg res weights from inhibitory pops
        """
        def __init__(self,SNN_dict):
                SNN_column.__init__(self,SNN_dict)

        def create_projections(self, n_in, n_res, n_readout, n_hli, w_in, w_res, w_out, Nexc, p_connect_sensor, p_connect_inter, w_intra_EE, w_intra_EI, w_intra_IE, interPop_delay):

                # =====================================================================================================
                # create projections
                # =====================================================================================================
                # projections_sensor_monitor
                for idx in range(n_in+n_hli):
                    conn_list = [(x, idx) for x in range(Nexc)]
                    connector = sim.FromListConnector(conn_list)
                    projection = sim.Projection(self.sensor_populations[idx], self.sensor_monitor_population, connector,
                                                sim.StaticSynapse(weight=1.0))

                # projections_sensor_hiddenP
                for idx0, pop in enumerate(self.hidden_populations):
                    if sum(w_in[idx0]) != 0:
                        for idx1 in range(n_in+n_hli):
                            projection_inp_hiddenP = sim.Projection(self.sensor_populations[idx1], pop[0],
                                                                    sim.FixedProbabilityConnector(p_connect=p_connect_sensor),
                                                                    sim.StaticSynapse(weight=w_in[idx0, idx1]))

                # projections_hiddenP_monitor
                for idx in range(n_res):
                    connector = sim.FromListConnector(conn_list=[(x, idx) for x in range(Nexc)])
                    projection = sim.Projection(self.hidden_populations[idx][0], self.monitor_population, connector,
                                                sim.StaticSynapse(weight=1.0))

                # projections_hiddenP_readout
                projections_hiddenP_readout = []
                connector = sim.AllToAllConnector()  # , delays=np.random.randint(6,72), rng=rng
                for idx0 in range(n_readout):
                    readout_pop = self.readout_neuron_populations[idx0]
                    projections = []
                    for idx1 in range(n_res):
                        pop0exc = self.hidden_populations[idx1][0]
                        projection = sim.Projection(pop0exc, readout_pop, connector,
                                                    sim.StaticSynapse(weight=w_out[idx0, idx1]))
                        projections.append(projection)
                    projections_hiddenP_readout.append(projections)

                # projections_hiddenP_hiddenP
                rng = sim.NumpyRNG(seed=2007200)
                for idx0 in range(n_res):
                        for idx1 in range(n_res):
                                W = float(w_res[idx1, idx0])
                                Pc = p_connect_inter[idx1, idx0]
                                if Pc > 0.001:
                                        pop1exc = self.hidden_populations[idx1][0] #target pop
                                        if W >0.001: #connection from excitatory
                                                pop0exc = self.hidden_populations[idx0][0]
                                                projection = sim.Projection(pop0exc, pop1exc,
                                                                            sim.FixedProbabilityConnector(p_connect=Pc,rng=rng),
                                                                            sim.StaticSynapse(weight=W, delay=float(interPop_delay[idx1, idx0])))
                                        if W <-0.001: #connection from inhibitory
                                                pop0inh = self.hidden_populations[idx0][1]
                                                projection = sim.Projection(pop0inh, pop1exc,
                                                                            sim.FixedProbabilityConnector(p_connect=Pc,rng=rng),
                                                                            sim.StaticSynapse(weight=W*4, delay=float(interPop_delay[idx1, idx0])),receptor_type='inhibitory')

                # create intrapopulation projections
                for idx in range(n_res):
                    hiddenPexc = self.hidden_populations[idx][0]
                    hiddenPinh = self.hidden_populations[idx][1]
                    projection_intra_EI = sim.Projection(hiddenPexc, hiddenPinh, self.connector_intra_EI,
                                                                      sim.StaticSynapse(weight=w_intra_EI))
                    projection_intra_EE = sim.Projection(hiddenPexc, hiddenPexc, self.connector_intra_EE,
                                                                      sim.StaticSynapse(weight=w_intra_EE))
                    projection_intra_IE = sim.Projection(hiddenPinh, hiddenPexc, self.connector_intra_IE,
                                                                      sim.StaticSynapse(weight=w_intra_IE),receptor_type='inhibitory')


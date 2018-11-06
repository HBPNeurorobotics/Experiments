from pyNN.random import NumpyRNG, RandomDistribution
import pyNN.nest as sim


class CellsParameters():

    @classmethod
    def motoneuronsStimWeight(cls):
        return 3

    @classmethod
    def motoneurons(cls,cellTypeName,rngSeed=123):
        rng = NumpyRNG(seed=rngSeed, parallel_safe=True)
        refractoryPeriod = RandomDistribution('normal', [20, 2], rng=rng)
        tauM = RandomDistribution('normal', [6, 0.6], rng=rng)
        if cellTypeName.lower() == "if_cond_alpha":
            parameters = {'cm': 0.25,
                         'e_rev_E': 0.0,
                         'e_rev_I': -70.0,
                         'i_offset': 0.0,
                         'tau_m': tauM,
                         'tau_refrac': refractoryPeriod,
                         'tau_syn_E': 0.25,
                         'tau_syn_I': 4.5,
                         'v_reset': -65.0,
                         'v_rest': -65.0,
                         'v_thresh': -50.0}
            cellType = sim.IF_cond_alpha(**parameters)
        elif cellTypeName.lower() == "if_curr_alpha":
            parameters = {'cm': 0.25,
                         'i_offset': 0.0,
                         'tau_m': tauM,
                         'tau_refrac': refractoryPeriod,
                         'tau_syn_E': 0.25,
                         'tau_syn_I': 4.5,
                         'v_reset': -65.0,
                         'v_rest': -65.0,
                         'v_thresh': -50.0}
            cellType = sim.IF_curr_alpha(**parameters)
        else: raise(Exception("Not valid cellTypeName"))
        return cellType

    @classmethod
    def interneurons(cls,cellTypeName,rngSeed=123):
        rng = NumpyRNG(seed=rngSeed, parallel_safe=True)
        refractoryPeriod = RandomDistribution('normal', [8, .8], rng=rng)
        tauM = RandomDistribution('normal', [30, 1], rng=rng)
        if cellTypeName.lower() == "if_cond_alpha":
            parameters = {'cm': 0.25,
                         'e_rev_E': 0.0,
                         'e_rev_I': -70.0,
                         'i_offset': 0.0,
                         'tau_m': tauM,
                         'tau_refrac': refractoryPeriod,
                         'tau_syn_E': 0.5,
                         'tau_syn_I': 10.,
                         'v_reset': -65.0,
                         'v_rest': -65.0,
                         'v_thresh': -50.0}
            cellType = sim.IF_cond_alpha(**parameters)
        elif cellTypeName.lower() == "if_curr_alpha":
            parameters = {'cm': 0.25,
                         'i_offset': 0.0,
                         'tau_m': tauM,
                         'tau_refrac': refractoryPeriod,
                         'tau_syn_E': 0.5,
                         'tau_syn_I': 10.,
                         'v_reset': -65.0,
                         'v_rest': -65.0,
                         'v_thresh': -50.0}
            cellType = sim.IF_curr_alpha(**parameters)
        else: raise(Exception("Not valid cellTypeName"))
        return cellType

    @classmethod
    def afferentFiber(cls,rngSeed=123):
        rng = NumpyRNG(seed=rngSeed, parallel_safe=True)
        refractoryPeriod = RandomDistribution('normal', [1000./700., (1000./700.)*0.05], rng=rng)
        tauM = .5
        parameters = {'cm': 0.25,
                     'i_offset': 0.0,
                     'tau_m': tauM,
                     'tau_refrac': refractoryPeriod,
                     'tau_syn_E': 0.5,
                     'tau_syn_I': 10.,
                     'v_reset': -65.0,
                     'v_rest': -65.0,
                     'v_thresh': -63.0}
        cellType = sim.IF_curr_alpha(**parameters)
        return cellType

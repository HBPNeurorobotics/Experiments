# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
@nrp.NeuronMonitor(nrp.brain.sensor_population0, nrp.spike_recorder)
def csv_spike_monitor(t):
        # Uncomment to log into the 'log-console' visible in the simulation
        # clientLogger.info("Time: "+str(t))
        return True

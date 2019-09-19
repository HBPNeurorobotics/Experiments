import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
from std_msgs.msg import Float64
@nrp.MapSpikeSink("agonist_neuron", nrp.brain.VN_agonist[slice(0,100,1)], nrp.population_rate, tau_fall = 20.0, tau_rise = 10.0)
@nrp.MapSpikeSink("antagonist_neuron", nrp.brain.VN_antagonist[slice(0,100,1)], nrp.population_rate, tau_fall = 20.0, tau_rise = 10.0)
#@nrp.MapSpikeSink("agonist_neuron", nrp.brain.VN_agonist[slice(0,100,1)], nrp.leaky_integrator_alpha, delay = 1.0, cm = 13.6)
#@nrp.MapSpikeSink("antagonist_neuron", nrp.brain.VN_antagonist[slice(0,100,1)], nrp.leaky_integrator_alpha, delay = 1.0, cm = 13.6)
@nrp.Neuron2Robot(Topic('/icub/eye_version/vel', Float64))
def eye_twist(t, agonist_neuron, antagonist_neuron):
    MAX_AMPLITUDE = 0.8
    eye_velocity=((agonist_neuron.rate - antagonist_neuron.rate)/(100.0*85.0))*MAX_AMPLITUDE*2.0*3.141592*1.55
        
    #clientLogger.info(t, agonist_neuron.rate, antagonist_neuron.rate, eye_velocity)
    if abs(eye_velocity)>MAX_AMPLITUDE*2.0*3.141592*1.55*1.2:
        sign=eye_velocity/(abs(eye_velocity))# + or -
        eye_velocity=MAX_AMPLITUDE*2.0*3.141592*1.5*1.2*sign   

    return std_msgs.msg.Float64(eye_velocity)

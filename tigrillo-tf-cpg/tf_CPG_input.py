# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_excontrol.logs import clientLogger
import generate_cpg_control as gcpg
from tigrillo_2_plugin.msg import Motors, Sensors
import rospy

mu = [504.035634768, 504.035634768, 1212.81284325, 1212.81284325] #amplitude?
o = [8.12457165988, 8.12457165988, -21.5669927808, -21.5669927808] #offset
omega = [6.28, 6.28, 6.28, 6.28 ]#frequencyish 8.966
d = [0.268511340639,0.268511340639,0.849917863287,0.849917863287]
phase_offset = [3.25231043641, 4.45153552476, 1.37346859255]
cpg = gcpg.CPGControl(mu, o, omega, d, phase_offset)

@nrp.MapVariable('cpg0',initial_value = cpg)
@nrp.MapVariable('pub',initial_value = rospy.Publisher("tigrillo_rob/uart_actuators", Motors, queue_size=1))

@nrp.Neuron2Robot()
def tf_CPG_input(t, pub, cpg0):
	#clientLogger.info(t)
	for i in range(20):
		next = cpg0.value.step_open_loop()
	pub.value.publish(run_time=t, FL=next[0], FR=next[1], BL=next[2], BR=next[3])
	return
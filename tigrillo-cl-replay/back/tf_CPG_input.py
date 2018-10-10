# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from std_msgs.msg import Float64
from hbp_nrp_excontrol.logs import clientLogger
from tigrillo_2_plugin.msg import Motors, Sensors
import rospy

import pandas as pd

motor_signal = np.array(pd.read_csv('resources/motor_signal_CL_ANN_walking_PostLearning.csv').iloc[1:, 1:])

@nrp.MapVariable('i',initial_value = 0)
@nrp.MapVariable('motor_signal',initial_value = motor_signal)
@nrp.MapVariable('pub',initial_value = rospy.Publisher("tigrillo_rob/uart_actuators", Motors, queue_size=1))


@nrp.Neuron2Robot()
def tf_CPG_input(t,i, pub, motor_signal):

	import time
	time.sleep(0.015)
	#clientLogger.info(t)
	ms = motor_signal.value
	pub.value.publish(run_time=t, FL=ms[i.value,0], FR=ms[i.value,1], BL=ms[i.value,2], BR=ms[i.value,3])
	i.value = i.value+1
    	return

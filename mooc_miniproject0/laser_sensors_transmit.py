import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import numpy as np
'''
Transmit laser scan information into sensory neuron activity:
The laser scan is composed of 360 beams, these beams are divided into 2 slices: right and left.
Sensory signals (laser values) are summed up and transformed by an activation function. 
The result is the rate of a Poisson generator that stimulates sensory neurons.
'''
@nrp.MapRobotSubscriber("laser", Topic("/robot/p3dx/laser/scan", sensor_msgs.msg.LaserScan))
@nrp.MapSpikeSource("right_sensor", nrp.brain.sensors[0], nrp.poisson)
@nrp.MapSpikeSource("left_sensor", nrp.brain.sensors[1], nrp.poisson)
@nrp.Robot2Neuron()
def laser_sensors_transmit(t, right_sensor, left_sensor, laser):   
    # Handles None value
    if laser.value is None: 
        left_sensor.rate = 100.
        return 0
    # Activation function definition
    def activation_fct(x, x_intercept, y_intercept):
        slope = - y_intercept / x_intercept
        return max(0, y_intercept + slope * x)   
    
    ##_TO_DO_## Change 'idx_right' to modify the visual angle involved in the sensory process
    idx_right = 0 # first right beam index
    idx_middle = 180 # index separating left and right beams
    idx_left = 360 - idx_right # last left beam index
    # Sum of laser beam values 
    # (using the mean enables to modify beam's slices without changing activation function parameterss)
    right_signal = np.mean(laser.value.ranges[slice(idx_right, idx_middle)])
    left_signal = np.mean(laser.value.ranges[slice(idx_middle, idx_left)])
    # Activation function arguments (axes intercepts)
    x_0 = 1. ##_TO_DO_##
    y_0 = 0. ##_TO_DO_##
    # Poisson generator rates
    right_sensor.rate = activation_fct(right_signal, x_0, y_0)
    left_sensor.rate = activation_fct(left_signal, x_0, y_0)
    # Print values in the log console
    if t % 2 < 0.02:
        clientLogger.info('[laser_sensors_transmit.py] right signal: {}, left signal: {}'.format(right_signal, left_signal))
        clientLogger.info('[laser_sensors_transmit.py] right sensor rate: {}, left sensor rate: {}'.format(right_sensor.rate, left_sensor.rate))
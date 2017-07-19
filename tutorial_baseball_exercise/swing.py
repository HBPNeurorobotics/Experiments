@nrp.MapSpikeSink("motors", nrp.brain.motors, nrp.leaky_integrator_alpha)
#################################################
# Insert code here:
# Add the joint to control "r_shoulder_yaw"
#################################################
@nrp.Neuron2Robot()
def swing(t, motors):
    # clientLogger.info("Motor potential: {}".format(motors.voltage))
    #################################################
    # Insert code here:
    # Control the joint "r_shoulder_yaw"
    #################################################
    pass

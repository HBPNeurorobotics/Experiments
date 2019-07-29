import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
from geometry_msgs.msg import Twist, Vector3
'''
Transform actor neurons activity into velocity commands
'''
@nrp.MapSpikeSink("left_actor", nrp.brain.actors[1], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("right_actor", nrp.brain.actors[0], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic("/robot/cmd_vel", geometry_msgs.msg.Twist))
def velocity_commands(t, right_actor, left_actor):
    # In abscence of activity, the vehicle is moving forward (magnitude lin_max)
    lin_max = 0.6
    # factor for the transformation from actor's voltage to linear translation component
    lin_factor = 120.
    # linear translation component obtained by substrackting actors voltage
    x_lin = max(0, lin_max - lin_factor*(right_actor.voltage + left_actor.voltage))
    # factor for the transformation from actor's voltage to angular rotation component
    ang_factor = 60.
    # angular rotation component as the actors voltage difference
    z_ang = ang_factor * (left_actor.voltage - right_actor.voltage)
    # print information on the log console
    if t % 2 < 0.02:
        clientLogger.info('linear x: {}, angular z: {}'.format(x_lin, z_ang))
    # build geometry messages
    Vlin = geometry_msgs.msg.Vector3(x_lin,0,0)
    Vang = geometry_msgs.msg.Vector3(0,0,z_ang)
    # send commands to wheels
    return geometry_msgs.msg.Twist(linear=Vlin, angular=Vang)
        
     
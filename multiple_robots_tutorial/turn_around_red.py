from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.Neuron2Robot(Topic('/red_pioneer3dx/cmd_vel', geometry_msgs.msg.Twist))
def turn_around_red_p3dx(t):
     return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(1.0,0,0),
                                     angular=geometry_msgs.msg.Vector3(0,0,0.25))
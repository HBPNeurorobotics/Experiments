from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.MapSpikeSink("red_fw", nrp.brain.red_fw[0], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("red_bw", nrp.brain.red_bw[0], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("red_left", nrp.brain.red_left[0], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("red_right", nrp.brain.red_right[0], nrp.leaky_integrator_alpha)
@nrp.Neuron2Robot(Topic('/red_pioneer3dx/cmd_vel', geometry_msgs.msg.Twist))
def turn_around_red_p3dx(t, red_fw, red_bw, red_left, red_right):
    return geometry_msgs.msg.Twist(linear=geometry_msgs.msg.Vector3(15 * red_fw.voltage if red_bw.voltage < 0.03 else  -8 * red_bw.voltage,0,0),
                                       angular=geometry_msgs.msg.Vector3(0,0, (100 * (red_right.voltage - red_left.voltage)) if (red_right.voltage > 0.5 or red_left.voltage > 0.5) else 0.02))
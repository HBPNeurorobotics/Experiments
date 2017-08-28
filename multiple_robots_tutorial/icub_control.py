# Imported Python Transfer Function
import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

# Imported Python Transfer Function
@nrp.MapSpikeSink("left_output_neuron", nrp.brain.icub_output_left[0], nrp.leaky_integrator_alpha)
@nrp.MapSpikeSink("right_output_neuron", nrp.brain.icub_output_right[0], nrp.leaky_integrator_alpha)
@nrp.MapRobotPublisher('l_shoulder_roll', Topic('/robot/l_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_pitch', Topic('/robot/l_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_shoulder_yaw', Topic('/robot/l_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('l_elbow', Topic('/robot/l_elbow/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_roll', Topic('/robot/r_shoulder_roll/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_pitch', Topic('/robot/r_shoulder_pitch/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_shoulder_yaw', Topic('/robot/r_shoulder_yaw/pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('r_elbow', Topic('/robot/r_elbow/pos', std_msgs.msg.Float64))
@nrp.MapVariable("left_leak", initial_value=1)
@nrp.MapVariable("right_leak", initial_value=1)
@nrp.Neuron2Robot()
def icub_control(t, left_output_neuron, right_output_neuron,
    l_shoulder_roll, l_shoulder_pitch, l_shoulder_yaw, l_elbow, 
    r_shoulder_roll, r_shoulder_pitch, r_shoulder_yaw, r_elbow, left_leak, right_leak):
    
    def wave_hand(side, roll, pitch, yaw, elbow):
        if side == 1:
            r_shoulder_roll.send_message(std_msgs.msg.Float64(roll))
            r_shoulder_pitch.send_message(std_msgs.msg.Float64(pitch))
            r_shoulder_yaw.send_message(std_msgs.msg.Float64(yaw))
            r_elbow.send_message(std_msgs.msg.Float64(elbow))
        elif side == -1:
            l_shoulder_roll.send_message(std_msgs.msg.Float64(roll))
            l_shoulder_pitch.send_message(std_msgs.msg.Float64(pitch))
            l_shoulder_yaw.send_message(std_msgs.msg.Float64(yaw))
            l_elbow.send_message(std_msgs.msg.Float64(elbow))

    def actuate_arm(side, voltage, leak):
        elbow = 1.5
        threshold = 0.02
        if voltage > threshold:
            wave_hand(side, 1.7, -1.1, 0., elbow)
            return 0.0
        else:
            leak = leak + 0.007
            if leak > 1.0:
                leak = 1.0
            r = 1 - leak
            wave_hand(side, 1.7 * r, -1.1 * r, 0., elbow)
            return leak
    
    left_leak.value = actuate_arm(-1, left_output_neuron.voltage, left_leak.value)
    right_leak.value = actuate_arm(1, right_output_neuron.voltage, right_leak.value)

# Imported Python Transfer Function
#
import gazebo_msgs.srv
from rospy import ServiceProxy, wait_for_service
from rospy import Duration
clientLogger.info('Waiting for ROS Service /gazebo/apply_joint_effort')
wait_for_service('/gazebo/apply_joint_effort')
clientLogger.info('Found ROS Service /gazebo/apply_joint_effort')
service_proxy = ServiceProxy('/gazebo/apply_joint_effort', gazebo_msgs.srv.ApplyJointEffort, persistent=True)
duration_val = Duration.from_sec(0.02)
@nrp.MapVariable("proxy", initial_value=service_proxy)
@nrp.MapVariable("duration", initial_value=duration_val)
@nrp.MapVariable("max_joint_force", initial_value=2.0)
@nrp.MapVariable("force_per_spike", initial_value=0.1)
@nrp.MapSpikeSink("proximal_extension", nrp.brain.index_proximal_extension_motor, nrp.population_rate)
@nrp.MapSpikeSink("proximal_contraction", nrp.brain.index_proximal_contraction_motor, nrp.population_rate)
@nrp.MapSpikeSink("distal_extension", nrp.brain.index_distal_extension_motor, nrp.population_rate)
@nrp.MapSpikeSink("distal_contraction", nrp.brain.index_distal_contraction_motor, nrp.population_rate)
@nrp.Neuron2Robot()
def motor_to_force_tf(t, proxy, duration,  max_joint_force, force_per_spike, proximal_extension, proximal_contraction, distal_extension, distal_contraction):
    try:
        # A place holder for a finger muscle model
        def finger_muscle_model(spike_rate, force_per_spike):
            # In this simple model every motor neuron applies a fixed amount of torque to an associated joint
            # The current joint state and muscle contraction dynamics are ignored for simplicity
            f = min(max_joint_force.value, spike_rate * force_per_spike)
            return f
        # Apply f_per_spike torque for d nanoseconds for each spike from a motor neuron
        d = duration.value
        f_per_spike = force_per_spike.value
        # Proximal Index Finger
        f_proximal_ext = -finger_muscle_model(proximal_extension.rate, f_per_spike)
        f_proximal_cont = finger_muscle_model(proximal_contraction.rate, f_per_spike)
        # Apply force to antagonist muscle
        proxy.value.call('hollie_real_left_hand_Index_Finger_Proximal', f_proximal_ext,  None, d)
        # Apply force to synergist muscle
        proxy.value.call('hollie_real_left_hand_Index_Finger_Proximal', f_proximal_cont,  None, d)
        # Distal Index Finger
        f_distal_ext = -finger_muscle_model(distal_extension.rate, f_per_spike)
        f_distal_cont = finger_muscle_model(distal_contraction.rate, f_per_spike)
        # Apply force to antagonist muscle
        proxy.value.call('hollie_real_left_hand_Index_Finger_Distal', f_distal_ext, None, d)
        # Apply force to synergist muscle
        proxy.value.call('hollie_real_left_hand_Index_Finger_Distal', f_distal_cont, None, d)
    except Exception as e:
        clientLogger.info(str(e))
#


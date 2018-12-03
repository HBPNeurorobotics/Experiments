from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates
@nrp.MapRobotPublisher('knee_jerk', Topic('/gazebo_muscle_interface/robot/vastus/cmd_activation', Float64))
@nrp.MapRobotSubscriber('muscle_states_msg', Topic('/gazebo_muscle_interface/robot/muscle_states', MuscleStates))
@nrp.Robot2Neuron()
def reflex_controller(t, knee_jerk, muscle_states_msg):
    # Muscle Properties
    m_optimal_fiber_length = 0.19
    m_max_contraction_velocity = 10.0
    
    # Get muscle state
    muscle_states =dict((m.name, m) for m in muscle_states_msg.value.muscles)
    
    # Muscle Lengthening speed
    m_speed = muscle_states['vastus'].lengthening_speed

    # Maximum muscle speed
    m_max_speed = m_optimal_fiber_length*m_max_contraction_velocity

    #:  Knee jerk reflex control

    # Reflex gain
    reflex_gain = 2.
    
    m_reflex_activation = min(1., 0.2*reflex_gain*(abs(m_speed) + m_speed)/m_max_speed)

    # Send muscle activation
    knee_jerk.send_message(m_reflex_activation)

# Imported Python Transfer Function
#
from gazebo_msgs.msg import ModelState
globals()['target_freq'] = 0.2
globals()['target_ampl'] = 0.6
globals()['target_center'] = {'x': 0, 'y': 2.42, 'z': 1.2}
@nrp.Neuron2Robot(Topic('/gazebo/set_model_state', ModelState))
def move_target(t):
    from gazebo_msgs.msg import ModelState
    m = ModelState()
    frequency = globals()['target_freq']
    amplitude = globals()['target_ampl']
    m.model_name = 'Target'
    # set orientation RYP axes
    m.pose.orientation.x = 0
    m.pose.orientation.y = 1
    m.pose.orientation.z = 1
    target_center = globals()['target_center']
    m.reference_frame = 'world'
    m.pose.position.x = \
        target_center['x'] + np.sin(t * frequency * 2 * np.pi) * (float(amplitude) / 2)
    m.pose.position.y = target_center['y']
    m.pose.position.z = target_center['z']
    m.scale.x = m.scale.y = m.scale.z = 1.0
    return m
#


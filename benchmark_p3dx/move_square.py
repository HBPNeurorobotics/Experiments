import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.MapRobotSubscriber("position", Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
# The motion is split in 8 steps:
# - step 0: move through the first side of the square
# - step 1: turn
# - step 2: move through the second side of the square
# - step 3: turn
# - step 4: move through the third side of the square
# - step 5: turn
# - step 6: move through the forth side of the square
# 'step_index' keeps track of the motion step the robot is currently applying
@nrp.MapVariable("step_index", global_key="step_index", initial_value=0)
@nrp.MapVariable("initial_pose", global_key="initial_pose", initial_value=None)
@nrp.Neuron2Robot(Topic('/p3dx/cmd_vel', geometry_msgs.msg.Twist))
def move_square(t, step_index, position, initial_pose):
    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('p3dx')].position
    linear = geometry_msgs.msg.Vector3(0,0,0)
    angular = geometry_msgs.msg.Vector3(0,0,0)
    current_pose = position.value.pose[position.value.name.index('p3dx')]
    import math
    cos45deg = math.cos(math.pi / 4)
    if step_index.value == 0:
        if current_pose.position.x < (initial_pose.value.x + 2):
            linear.x = 0.2
        else:
            step_index.value = 1
    elif step_index.value == 1:
        if current_pose.orientation.z > (-1 * cos45deg):
            angular.z = -0.2
        else:
            step_index.value = 2
    elif step_index.value == 2:
        if current_pose.position.y > (initial_pose.value.y - 2):
            linear.x = 0.2
        else:
            step_index.value = 3
    elif step_index.value == 3:
        if current_pose.orientation.z > -1 and current_pose.orientation.w > 0:
            angular.z = -0.2
        else:
            step_index.value = 4
    elif step_index.value == 4:
        if current_pose.position.x > (initial_pose.value.x):
            linear.x = 0.2
        else:
            step_index.value = 5
    elif step_index.value == 5:
        if current_pose.orientation.w < cos45deg and current_pose.orientation.z < (-1 * cos45deg):
            angular.z = -0.2
        else:
            step_index.value = 6
    elif step_index.value == 6:
        if current_pose.position.y < initial_pose.value.y:
            linear.x = 0.2
        else:
            step_index.value = 7
    return geometry_msgs.msg.Twist(linear=linear,angular=angular)

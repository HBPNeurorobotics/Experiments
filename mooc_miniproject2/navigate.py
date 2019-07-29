import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg

@nrp.MapRobotSubscriber("pose", Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapVariable("step_index", global_key="step_index", initial_value=0)
@nrp.MapVariable("initial_position", global_key="initial_position", initial_value=None)
@nrp.MapVariable("initial_orientation", global_key="initial_orientation", initial_value=None)
@nrp.MapVariable("waypoints", global_key="waypoints", initial_value=None)
@nrp.MapVariable("lattice_points", global_key="lattice_points", initial_value=None)
@nrp.MapVariable("stage", global_key="stage", initial_value=1)
@nrp.Neuron2Robot(Topic('/robot/cmd_vel', geometry_msgs.msg.Twist))
def navigate(t, step_index, pose, initial_position, initial_orientation, waypoints, lattice_points, stage):
    import math; import numpy; import csv
    from load_csv_positions import load_waypoints, load_lattice
    if lattice_points.value is None:
        # Load data of the SOM-lattice from a csv file 
        lattice_points.value = load_lattice("resources/lattice.csv")

    if waypoints.value is None:
        # Load data of the way points from a csv file 
        waypoints.value = load_waypoints("resources/waypoints.csv")
        
    if initial_position.value is None:
        initial_position.value = pose.value.pose[pose.value.name.index('robot')].position
        initial_orientation.value = pose.value.pose[pose.value.name.index('robot')].orientation
    
    Nw = waypoints.value.shape[0]
    if (stage.value < Nw) and (t > 5):
        current_pose = pose.value.pose[pose.value.name.index('robot')]
        current_orientation = current_pose.orientation
        current_position = current_pose.position
        p_robot = [ current_position.x, current_position.y ]
        x = waypoints.value[stage.value, 0]
        y = waypoints.value[stage.value, 1]
        p_target = [lattice_points.value[x, y, 0.0], lattice_points.value[x, y, 1.0]]
        delta_x = p_target[0] - p_robot[0]
        delta_y = p_target[1] - p_robot[1]
        norm = math.sqrt(delta_x**2 + delta_y**2)
        if norm != 0:
            delta_x = delta_x / norm
            delta_y = delta_y / norm
        Delta = math.atan2(delta_y, delta_x)
        d = 2.0 * math.atan2(current_orientation.z, current_orientation.w)
        linear = geometry_msgs.msg.Vector3(0.0, 0.0, 0.0)
        angular = geometry_msgs.msg.Vector3(0.0, 0.0, 0.0)
    
        epsilon = 0.1
        diff = Delta - d
        abs_diff = math.fabs(diff)
    
        # Move forward only if the directions are close enough
        if abs_diff < epsilon:
            linear.x = 0.5
            angular.z = 0.0
        else:
            linear.x = 0.0
    
        # Change the direction of rotation depending of the direction
        # difference
        rotation_coeff = 1.0
        if diff > epsilon:
            angular.z = rotation_coeff * abs_diff
        if diff < - epsilon:
            angular.z = - rotation_coeff * abs_diff
    
        if norm < 0.05 or stage.value > Nw:
            linear = geometry_msgs.msg.Vector3(0.0, 0.0, 0.0)
            angular = geometry_msgs.msg.Vector3(0.0, 0.0, 0.0)

        if norm < 0.05:
            stage.value = stage.value + 1
        
        if stage.value > Nw:
            stage.value = stage.value - 1
        
        return geometry_msgs.msg.Twist(linear=linear,angular=angular)
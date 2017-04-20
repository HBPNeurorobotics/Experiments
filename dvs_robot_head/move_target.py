#!/usr/bin/env python
"""
"""

__author__ = 'Jacques Kaiser'

import logging
import time
import rospy
import random
import smach_ros
from smach.state import State
from std_msgs.msg import String
from smach import StateMachine

from geometry_msgs.msg import Wrench, Vector3, Point

from rospy import ServiceProxy, wait_for_service
from gazebo_msgs.srv import ApplyBodyWrench, GetModelState


FINISHED = 'FINISHED'
ERROR = 'ERROR'
PREEMPTED = 'PREEMPTED'

sm = StateMachine(outcomes=[FINISHED, ERROR, PREEMPTED])
clientLogger = rospy.Publisher('/ros_cle_simulation/logs', String, latch=True,
                               queue_size=10)

class GenerateRandomTargetForceState(State):
    def __init__(self, model_name,
                 outcomes=['success', 'aborted']):
        super(GenerateRandomTargetForceState, self).__init__(outcomes=outcomes)

        clientLogger.publish('Waiting for ROS Services')
        wait_for_service('/gazebo/apply_body_wrench')
        wait_for_service('/gazebo/get_model_state')
        clientLogger.publish('Found ROS Services')
        self.wrench_proxy = ServiceProxy('/gazebo/apply_body_wrench',
                                         ApplyBodyWrench, persistent=True)
        self.state_proxy = ServiceProxy('/gazebo/get_model_state',
                                        GetModelState, persistent=True)

        self.model_name = model_name
        self.sign = 1
        self._seed = random.seed(None)
        clientLogger.publish("GenerateRandomTargetForceState model_name={}"
                             .format(model_name))

    def execute(self, userdata):
        current_target_state = self.state_proxy(self.model_name, "world")
        current_x = current_target_state.pose.position.x
        # bias to keep the ball centered in x=0
        force_x = self.sign * 30
        self.sign = - self.sign
        force = Vector3(force_x, 0, 25)
        wrench = Wrench(force, Vector3(0.0, 0.0, 0.0))

        self.wrench_proxy(self.model_name, "world", Point(), wrench,
                           rospy.Time(0.0), rospy.Duration(0.1))


        clientLogger.publish("New force applied: {} {} {}"
                             .format(force.x, force.y, force.z))
        return 'success'


class SleepState(State):
    def __init__(self, duration, outcomes=['success', 'aborted']):
        super(SleepState, self).__init__(outcomes=outcomes)
        self._duration = duration
        clientLogger.publish("SleepState with duration={duration}\
                             initialized".format(duration=duration))

    def execute(self, userdata):
        rospy.sleep(self._duration)
        return 'success'


with sm:
    StateMachine.add(
        "generate_target_force",
        GenerateRandomTargetForceState('target_sphere::link'),
        transitions = {'success': 'sleep_state',
                       'aborted': ERROR,}
    )
    StateMachine.add(
        "sleep_state",
        SleepState(2.0),
        transitions = {'success': 'generate_target_force',
                       'aborted': ERROR,}
    )

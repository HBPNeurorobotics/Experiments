#!/usr/bin/env python
"""
"""

__author__ = 'Igor Peric, Martin Schulze'

import logging
import time
import rospy
import cmath
import random
import smach_ros
from smach.state import State
from std_msgs.msg import Float64, String
from smach import StateMachine

FINISHED = 'FINISHED'
ERROR = 'ERROR'
PREEMPTED = 'PREEMPTED'

sm = StateMachine(outcomes=[FINISHED, ERROR, PREEMPTED])
clientLogger = rospy.Publisher('/ros_cle_simulation/logs', String, latch=True,
                               queue_size=10)

class GenerateRandomFingerMovementState(State):
    def __init__(self, ros_topic_name, min_angle, max_angle,
                 outcomes=['success', 'aborted']):
        super(GenerateRandomFingerMovementState, self).__init__(outcomes=outcomes)
        self._pub = rospy.Publisher(ros_topic_name, Float64, latch=True, queue_size=1)
        self._seed = random.seed(None)
        self._min_angle = min_angle
        self._max_angle = max_angle
        clientLogger.publish("GenerateRandomFingerMovementState topic={topic},\
                             min_angle={min_angle},\
                             max_angle={max_angle}".format(topic=ros_topic_name,
                             min_angle=min_angle,
                             max_angle=max_angle))

    def execute(self, userdata):
        angle = random.uniform(self._min_angle, self._max_angle)
        self._pub.publish(Float64(angle))
        clientLogger.publish("New target angle generated: {}".format(angle))
        return 'success'


class SleepState(State):
    def __init__(self, duration, outcomes=['success', 'aborted']):
        super(SleepState, self).__init__(outcomes=outcomes)
        self._duration = duration
        clientLogger.publish("SleepState with duration={duration}\
                             initialized".format(duration=duration))

    def execute(self, userdata):
        time.sleep(self._duration)
        return 'success'


with sm:
    StateMachine.add(
        "generate_index_finger_proximal_angle",
        GenerateRandomFingerMovementState('/target_angle_index_finger_proximal',
                                          0.0, 1.0),
        transitions = {'success': 'generate_index_finger_distal_angle',
                       'aborted': ERROR,}
    )
    StateMachine.add(
        "generate_index_finger_distal_angle",
        GenerateRandomFingerMovementState('/target_angle_index_finger_distal',
                                          0.0, 1.0),
        transitions = {'success': 'sleep_state',
                       'aborted': ERROR,}
    )
    StateMachine.add(
        "sleep_state",
        SleepState(5.0),
        transitions = {'success': 'generate_index_finger_proximal_angle',
                       'aborted': ERROR,}
    )

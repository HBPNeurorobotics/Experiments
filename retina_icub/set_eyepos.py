    # Imported Python Transfer Function
    #
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64
    @nrp.MapVariable("eye_position", initial_value=0, scope=nrp.GLOBAL)
    @nrp.MapVariable("eye_velocity", initial_value=0, scope=nrp.GLOBAL)
    @nrp.MapRobotSubscriber("joints", Topic("/robot/joints", JointState))
    @nrp.MapRobotSubscriber("eye_vel", Topic("/robot/eye_version/vel", Float64))
    @nrp.Robot2Neuron()
    def set_eyepos(t, eye_position, eye_velocity, joints, eye_vel):
        joints = joints.value
        eye_position.value = joints.position[joints.name.index('eye_version')]
        if eye_vel.value is not None:
            eye_velocity.value = eye_vel.value.data
    #

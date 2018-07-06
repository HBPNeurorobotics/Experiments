    # Imported Python Transfer Function
    #
    @nrp.MapVariable("old_joint_angle", scope=nrp.GLOBAL)
    @nrp.MapVariable("joint_angle", scope=nrp.GLOBAL)
    @nrp.MapVariable("is_random_movement", scope=nrp.GLOBAL)
    @nrp.MapVariable("topic_index", initial_value=-1)
    @nrp.MapSpikeSink("motor_contract", nrp.brain.actors[0], nrp.spike_recorder)
    @nrp.MapSpikeSink("motor_extend", nrp.brain.actors[1], nrp.spike_recorder)
    @nrp.MapSpikeSource("teaching_motor_contract", nrp.brain.actors[0], nrp.poisson, weight=100.0)
    @nrp.MapSpikeSource("teaching_motor_extend", nrp.brain.actors[1], nrp.poisson, weight=100.0)
    @nrp.MapRobotPublisher("topic_arm", Topic('/robot/hollie_real_left_arm_2_joint/cmd_pos', std_msgs.msg.Float64))
    @nrp.MapRobotSubscriber("topic_joint_states", Topic('/joint_states', sensor_msgs.msg.JointState))
    @nrp.Neuron2Robot()
    def propagate_motor_commands(t, old_joint_angle, joint_angle, is_random_movement, topic_index, motor_contract, motor_extend, teaching_motor_contract, teaching_motor_extend, topic_arm, topic_joint_states):
        try:
            import random
            movement_step = 0.5
            random_chance = 1.0
            teaching_rate = 0.0
            if not topic_joint_states.value:
                clientLogger.info("ROS topic /joint_states not found, skipping TF: propagate_motor_commands")
                return
            if topic_index.value == -1:
                topic_index.value = topic_joint_states.value.name.index('hollie_real_left_arm_2_joint')
            old_joint_angle.value = topic_joint_states.value.position[topic_index.value]
            if is_random_movement or random.uniform(0,1) > random_chance:
                rnd_num = random.uniform(0, 1)
                joint_delta = movement_step if rnd_num <= 0.5 else -movement_step
            else:
                if motor_extend.spiked and not motor_contract.spiked:
                    joint_delta = -movement_step
                    teaching_motor_extend.rate = teaching_rate
                    teaching_motor_contract.rate = 0.0
                elif motor_contract.spiked and not motor_extend.spiked:
                    joint_delta = movement_step
                    teaching_motor_contract.rate = teaching_rate
                    teaching_motor_extend.rate = 0.0
                else:
                    teaching_motor_extend.rate = 0.0
                    teaching_motor_contract.rate = 0.0
                    return
            if joint_delta < 0.0:
                joint_delta = joint_delta*1.5
            # Clip joint angle to joint limits
            joint_angle.value = min(3.14/2.0 , max(0, joint_delta + old_joint_angle.value))
            topic_arm.send_message(std_msgs.msg.Float64(joint_angle.value))
        except Exception as e:
            clientLogger.info(str(e))
    #

# Imported Python Transfer Function
#
@nrp.MapVariable("old_joint_angle", scope=nrp.GLOBAL)
@nrp.MapVariable("joint_angle", scope=nrp.GLOBAL)
@nrp.MapVariable("target_angle", scope=nrp.GLOBAL)
@nrp.MapSpikeSource("dopamine_source_increase", nrp.brain.dopamine[1], nrp.poisson, weight=100.0)
@nrp.MapSpikeSource("dopamine_source_decrease", nrp.brain.dopamine[0], nrp.poisson, weight=100.0)
@nrp.Robot2Neuron()
def trigger_dopamine(t, old_joint_angle, joint_angle, target_angle, dopamine_source_increase, dopamine_source_decrease):
    try:
        if joint_angle.value and old_joint_angle.value:
            is_reward_positive = abs(joint_angle.value - target_angle.value) < abs(old_joint_angle.value - target_angle.value)
            if is_reward_positive:
                dopamine_source_increase.rate = 10.0
                dopamine_source_decrease.rate = 0.0
            else:
                dopamine_source_increase.rate = 0.0
                dopamine_source_decrease.rate = 10.0
    except Exception as e:
        clientLogger.info(str(e))
#


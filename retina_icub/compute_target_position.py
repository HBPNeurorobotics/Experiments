    # Imported Python Transfer Function
    #
    # Compute Target position following a random linear trajectory
    @nrp.MapVariable("target_delta", scope=nrp.GLOBAL)
    @nrp.MapVariable("direction", initial_value=1)
    @nrp.MapVariable("counter", initial_value=0)
    @nrp.MapVariable("trajectory", scope=nrp.GLOBAL)
    @nrp.Robot2Neuron()
    def compute_target_position(t, target_delta, direction, counter, trajectory):
        if trajectory != "random_linear":
            return
        import random
        tf = hbp_nrp_cle.tf_framework.tf_lib
        counter.value = counter.value+ 1
        if counter.value == 50:
            counter.value = 0
            r = random.random()
            if r > 0.5:
                direction.value = -1 * direction.value
        if target_delta.value <= -.3 and direction.value == -1:
            direction.value = 1
        if target_delta.value >= .3 and direction.value == 1:
            direction.value = -1
        target_delta.value = target_delta.value + direction.value * 0.005
    #

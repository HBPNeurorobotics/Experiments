# Imported transfer function that deals with all the logic of the experiment
@nrp.MapVariable(   'isBlinking',  scope=nrp.GLOBAL, initial_value=False)
@nrp.MapSpikeSource('resetSignal', nrp.map_neurons(range(1), lambda i: nrp.brain.someNeuron[i]), nrp.dc_source, amplitude=0.0)
@nrp.Robot2Neuron()
def some_TF1(t, isBlinking, resetSignal):
    
    # Loop timing (stimulus onset, grouping trigger, selection signals, blinking)
    loopDuration     = 0.50    # units: seconds, duration of a blinking loop (starts with blink)
    blinkDuration    = 0.10    # units: seconds, how much time a blink lasts
    tTrue            = t-0.04  # mini de-synchronization with state machine

    # Initialization
    isBlinking.value = False

    # Time-related variables
    loopCount        = int(tTrue/loopDuration)
    lastLoopTime     = loopCount*loopDuration
    timeInLoop       = tTrue-lastLoopTime

    # Choose if this is a normal or a blinking time-step
    if 0.0 <= timeInLoop < blinkDuration:
        isBlinking.value    = True

    # Deal with the reset signal
    if isBlinking.value:
        resetSignal.amplitude = 1.0
    else:
        resetSignal.amplitude = 0.0
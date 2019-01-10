@nrp.MapVariable("msg_counter", initial_value=None)
@nrp.Robot2Neuron()
def documentation(t, msg_counter):
    if not msg_counter.value:
        msg_counter.value = 0
    if t > 0 and msg_counter.value == 0:
        clientLogger.advertise("This experiment demonstrates the integration of the Nengo simulator into the NRP.")
        msg_counter.value = 1
    elif t > 5 and msg_counter.value == 1:
        clientLogger.advertise("We use Transfer Functions to generate plots similar to the ones in Nengo GUI.")
        msg_counter.value = 2
    elif t > 10 and msg_counter.value == 2:
        clientLogger.advertise("The following plots can be displayed using the Video Stream view:")
        msg_counter.value = 3
    elif t > 15 and msg_counter.value == 3:
        clientLogger.advertise("/brain/ensembles_dimensions_plot displays an oscilloscope view for all nengo ensembles in the brain")
        msg_counter.value = 4
    elif t > 20 and msg_counter.value == 4:
        clientLogger.advertise("The plot can be customized in the Transfer Function ensembles_dimensions_plot")
        msg_counter.value = 5
    elif t > 25 and msg_counter.value == 5:
        clientLogger.advertise("The Transfer Function robot control plot displays the current neural output in a 2D plot.")
        msg_counter.value = 6
    elif t > 30 and msg_counter.value == 6:
        clientLogger.advertise("You can further increase the simulation performance by deactivating the plotting Transfer Functions.")
        msg_counter.value = 7
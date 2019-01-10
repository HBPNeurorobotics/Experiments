@nrp.MapSpikeSink("sensors", nrp.brain.sensors, nrp.raw_signal)
@nrp.MapSpikeSink("actors", nrp.brain.actors, nrp.raw_signal)
@nrp.MapVariable("figure", initial_value=None)
@nrp.MapVariable("lines")
@nrp.MapVariable("values")
@nrp.MapVariable("times", initial_value=None)
@nrp.MapVariable("last_publish", initial_value=-100.0)
@nrp.Neuron2Robot(Topic("/brain/ensembles_dimensions_plot", sensor_msgs.msg.Image))
def ensembles_dimensions_plot(t, sensors, actors, figure, lines, values, times, last_publish):
    # Edit the following parameters to customize the plot
    ensembles = {sensors: "Sensors", actors: "Actors"} # Which ensembles should be plotted
    publish_rate = 5  # How many plots per second should be generated
    storage_size = 1000  # Amount of historical values to store per dimension
    y_scale = [0.0, 1.1]  # Lower and upper limit of plot's y scale

    import numpy as np
    from cv_bridge import CvBridge

    def initialize():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from collections import deque

        figure.value = plt.figure(facecolor="0.975")

        num_subplots = len(ensembles)
        lines.value = []
        values.value = []
        for i, e in enumerate(ensembles):
            ax = figure.value.add_subplot(1, num_subplots, i + 1)
            ax.set_ylim(y_scale)
            ax.legend(loc='upper left', frameon=False)
            values.value.append(list(deque([], maxlen=storage_size) for _ in range(len(e.value))))
            lines.value.append(list(ax.plot(d, label='{}[{}]'.format(ensembles[e], j))[0] for j, d in enumerate(values.value[-1])))
        times.value = deque([], maxlen=storage_size)

    def update():
        times.value.append(t)
        for i, e in enumerate(ensembles):
            val_e = e.value
            for j, v in enumerate(val_e):
                values.value[i][j].append(v)

    def redraw():
        for i, e in enumerate(ensembles):
            for j, v in enumerate(e.value):
                lines.value[i][j].set_data(times.value, values.value[i][j])
        for ax in figure.value.get_axes():
            ax.set_xlim([times.value[0], t])
            ax.legend(loc='upper left', frameon=False)
        figure.value.canvas.draw()

    if not figure.value:
        initialize()
    else:
        update()

        if last_publish.value + 1.0 / publish_rate < t:
            redraw()
            arr = np.array(getattr(figure.value.canvas.renderer, '_renderer'))
            last_publish.value = t
            return CvBridge().cv2_to_imgmsg(arr, 'rgba8')

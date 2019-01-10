@nrp.MapSpikeSink("actors", nrp.brain.actors, nrp.raw_signal)
@nrp.MapVariable("figure", initial_value=None)
@nrp.MapVariable("last_publish", initial_value=-100.0)
@nrp.Neuron2Robot(Topic("/brain/robot_control_plot", sensor_msgs.msg.Image))
def robot_control_plot(t, actors, figure, last_publish):
    publish_rate = 5  # How many plots per second should be generated
    x_scale = [-1.1, 1.1]  # Lower and upper limit of plot's x scale
    y_scale = [-0.1, 1.1]  # Lower and upper limit of plot's y scale

    def initialize():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        figure.value = plt.figure(facecolor="0.975")
        ax = figure.value.add_subplot(111)
        ax.autoscale(enable=False)
        ax.set_ylim(y_scale)
        ax.set_xlim(x_scale)

    def redraw():
        ax = figure.value.get_axes()[0]
        ax.clear()
        ax.autoscale(enable=False)
        ax.set_ylim(y_scale)
        ax.set_xlim(x_scale)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.invert_xaxis()
        ax.set_xlabel("turn")
        ax.set_ylabel("drive forward")
        ax.scatter(actors.value[1], actors.value[0])
        figure.value.canvas.draw()

    if not figure.value:
        initialize()
    else:
        if last_publish.value + 1.0 / publish_rate < t:
            redraw()
            arr = np.array(getattr(figure.value.canvas.renderer, '_renderer'))
            last_publish.value = t
            return CvBridge().cv2_to_imgmsg(arr, 'rgba8')


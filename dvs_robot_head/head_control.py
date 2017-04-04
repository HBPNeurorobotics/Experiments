# Imported Python Transfer Function
import numpy as np
from sensor_msgs.msg import JointState
@nrp.MapSpikeSink("motor_recorder", nrp.brain.motors, nrp.spike_recorder)
@nrp.MapRobotPublisher('output_layer_plot', Topic('/output_layer', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher('head_pan', Topic('/robot/RHM0_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotPublisher('head_tilt', Topic('/robot/SwingXAxis_joint/cmd_pos', std_msgs.msg.Float64))
@nrp.MapRobotSubscriber("joint_states_msg", Topic('/joint_states', JointState))
@nrp.Neuron2Robot()
def head_control(t, output_layer_plot, head_pan, head_tilt, motor_recorder, joint_states_msg):
    if joint_states_msg.value is None:
        return

    # we count the spikes of each motor neurons
    output_shape = np.array(nrp.config.brain_root.output_shape)
    spike_array = np.zeros(output_shape[0] *
                           output_shape[1])
    spike_times = motor_recorder.times.tolist()
    if len(spike_times) == 0:
        return
    min_idx = int(motor_recorder.neurons[0])
    for (idx, time) in spike_times:
        idx = idx - min_idx
        spike_array[idx] = spike_array[idx] + 1

    center_idx = (output_shape / 2).astype(int)
    # reshape the 1D spike count to 2D
    spike_array = spike_array.reshape(output_shape[0],
                                          output_shape[1])

    # plot the activity of the output layer in 2D
    output_activity = spike_array * 255. / np.amax(spike_array)
    msg_frame = CvBridge().cv2_to_imgmsg(output_activity.astype(np.uint8), 'mono8')
    output_layer_plot.send_message(msg_frame)

    # get the motor neuron that spiked most
    max_motor = np.unravel_index(spike_array.argmax(), output_shape)

    # get the pan and tilt motor command
    hfov = 1.047
    vfov = hfov
    pan_control_map = np.linspace(hfov/2, -hfov/2, output_shape[1], endpoint = True)
    tilt_control_map = np.linspace(-vfov/2, vfov/2, output_shape[0], endpoint = True)

    if t < 0.2:
        clientLogger.info("Control map: {}".format(pan_control_map))

    delta_pan = pan_control_map[max_motor[1]]
    delta_tilt = tilt_control_map[max_motor[0]]

    # get current pan and tilt
    joint_states = joint_states_msg.value
    pan_idx = joint_states.name.index("RHM0_joint")
    current_pan = joint_states.position[pan_idx]
    tilt_idx = joint_states.name.index("SwingXAxis_joint")
    current_tilt = joint_states.position[tilt_idx]

    clientLogger.info("Pan and tilt:\ndeltas: {} {}\ncurrent: {} {}".format(delta_pan,
                                                                            delta_tilt,
                                                                            current_pan,
                                                                            current_tilt
                                                                        ))

    head_pan.send_message(std_msgs.msg.Float64(delta_pan + current_pan))
    head_tilt.send_message(std_msgs.msg.Float64(delta_tilt + current_tilt))

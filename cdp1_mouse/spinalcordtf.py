#from hbp_nrp_excontrol.logs import clientLogger
#from std_msgs.msg import Header

from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates
@nrp.MapVariable('clm', initial_value=None)
@nrp.MapVariable('runningSimulation', initial_value=None)
@nrp.MapRobotPublisher("activateFoot1", Topic("/gazebo_muscle_interface/robot/Foot1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateFoot2", Topic("/gazebo_muscle_interface/robot/Foot2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius1", Topic("/gazebo_muscle_interface/robot/Radius1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateRadius2", Topic("/gazebo_muscle_interface/robot/Radius2/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus1", Topic("/gazebo_muscle_interface/robot/Humerus1/cmd_activation", Float64))
@nrp.MapRobotPublisher("activateHumerus2", Topic("/gazebo_muscle_interface/robot/Humerus2/cmd_activation", Float64))
@nrp.MapRobotSubscriber("muscle_states_msg", Topic("/gazebo_muscle_interface/robot/muscle_states", MuscleStates))
@nrp.MapRobotSubscriber("joint_state_msg", Topic('/joint_states', JointState))
@nrp.MapRobotPublisher("spindlePlot", Topic("/mouse_locomotion/muscle_spindles_network", Image))
@nrp.Robot2Neuron()
def transferfunction( t, clm, runningSimulation,
                      activateFoot1, activateFoot2,
                      activateRadius1, activateRadius2,
                      activateHumerus1, activateHumerus2,
                      muscle_states_msg,
                      joint_state_msg,
                      spindlePlot):
    import traceback
    import time
    try:
        def get_sled_position_from_joint_states(joint_state_msg):
            joint_name = "cdp1_msled::world_sled"
            for name, position in zip(joint_state_msg.name, joint_state_msg.position):
                if name == joint_name:
                    # print "Found my joint: " + str(name) + " pos =" + str(position)
                    return position
            return None
        # Obtained by my printMuscleLengthBounds.py script.
        length_bounds = {
            'Foot1': (0.008153, 0.009358),
            'Foot2': (0.005556, 0.005871),
            'Radius1': (0.005420, 0.006850),
            'Radius2': (0.004728, 0.006186),
            'Humerus1': (0.013028, 0.013279),
            'Humerus2': (0.006273, 0.007500),
        }
        if clm.value is None:
            import sys
            import os
            clientLogger.info("Initializing Spinal Cord")
            path_to_code = os.path.join(os.environ['HBP'],'Experiments','cdp1_mouse','code')
            sys.path.append(path_to_code)
            os.environ['NN_CODE'] = path_to_code
            # WARNING: Any code changes in imported modules after initial import are ignored.
            # And that counts since the launch of the plattform .
            # NOTE: The simulations module is imported from Shravan's MouseLocomotion project!
            # Its source files reside in the neuralnetwork/code folder.
            import simulations
            eesFreq = 0.001
            eesAmp = 1
            species = "mouse"
            figName  = "videoNetwrokActivity"
            # The nnStructFile defines the NN configuration.
            # Network in 'closedLoopMouse.txt' is designed to actuate the two muscles
            # of the hind-foot in such a way that the hind foot tilts back and forth like a pendulum.
            nnStructFile = "closedLoopMouse.txt"
            # Instantiate the NN controller and internal transfer functions.
            # Note  that the internal transfer functions are derived from experimental data
            # of the hind limb. See MouseLocomotion/neuralnetwork/code/tools/afferents_tools.py
            clm.value = simulations.ClosedLoopSimulation(nnStructFile, species , eesAmp , eesFreq, figName)
            runningSimulation = True
            clientLogger.info("Initializing Spinal Cord - Done")
        else:
            muscle_states = dict((m.name, m) for m in muscle_states_msg.value.muscles)
            # Prepare argument dict for input to the NN.
            mmData = {'t':t*1000+20, 'stretch':{}}
            # The following are reference lengths for muscles used in NN control.
            # These numbers are approximately the initial lengths at t=0.
            # Obtain from rostopic echo for instance.
            normalized_lengths = {}
            for k, (a, b) in length_bounds.items():
                normalized_lengths[k] = (muscle_states[k].length - a)/(b - a)

            # Note: Input neuron fire rates increase with stretch!
            l_CE_CF = normalized_lengths['Humerus2'] * 0.4 + 0.0056
            l_CE_PMA = normalized_lengths['Humerus1'] * 0. + 0.0056
            l_CE_POP = normalized_lengths['Radius2'] * 0.4 + 0.0056
            l_CE_RF  = normalized_lengths['Radius1'] * 0. + 0.0056
            l_CE_TA  = normalized_lengths['Foot1'] * 0.4 + 0.0056
            l_CE_LG = normalized_lengths['Foot2'] * 0. + 0.0056

            # l_CE_SM = 0.0165
            # l_CE_POP = 0.00206
            # l_CE_RF = 0.00534
            # l_CE_TA = 0.0049
            # l_CE_SOL = 0.00316
            # l_CE_LG = 0.00541

            # The names l_CE_LG and l_CE_TA refer to the meaning of the corresponding
            # values in the hind-limb experiment, namely the lengths of the contractile
            # elements of the 'LG' and 'TA' muscles.
            mmData['stretch']['LEFT_PMA'] = l_CE_PMA
            mmData['stretch']['LEFT_CF'] = l_CE_CF
            mmData['stretch']['LEFT_POP'] = l_CE_POP
            mmData['stretch']['LEFT_RF'] = l_CE_RF
            mmData['stretch']['LEFT_LG'] = l_CE_LG
            mmData['stretch']['LEFT_TA'] = l_CE_TA

            # Advance the neural simulation.
            nnData = clm.value.run_step(mmData)

            def send_activations(humerus1, foot1, radius1, humerus2, foot2, radius2):
                for sender, value in [
                    (activateHumerus1, humerus1),
                    (activateFoot1, foot1),
                    (activateRadius1, radius1),
                    (activateHumerus2, humerus2),
                    (activateFoot2, foot2),
                    (activateRadius2, radius2)
                ]:
                    sender.send_message(value)
            send_activations(humerus1 = nnData['LEFT_PMA'],
                             foot1 = nnData['LEFT_TA'],
                             radius1 = nnData['LEFT_RF'],
                             humerus2 = nnData['LEFT_CF'],
                             foot2 = nnData['LEFT_LG'],
                             radius2 = nnData['LEFT_POP'])
            #Finally log some info.
            #clientLogger.info("-------------------------------")
            #clientLogger.info("Activations="+str(nnData))
            #clientLogger.info("Length=%f,%f" % (l_NORM_HUMERUS1, l_NORM_HUMERUS2))
            #clientLogger.info("Stretches=LG=%f, TA=%f" % (mmData['stretch']['LEFT_LG'], mmData['stretch']#['LEFT_TA']))
            #clientLogger.info("Forces="+str([m.force for m in muscle_states.values()]))
    except:
        tb = traceback.format_exc()
        clientLogger.info("Exception occured in spinal cord run_step: " + str(tb))
        time.sleep(1.)


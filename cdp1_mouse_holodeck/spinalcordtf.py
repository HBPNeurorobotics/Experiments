from std_msgs.msg import Float64
from gazebo_ros_muscle_interface.msg import MuscleStates
@nrp.MapVariable('clm', initial_value=None)
@nrp.MapRobotPublisher('activateFoot1',Topic('/gazebo_muscle_interface/robot/Foot1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateFoot2',Topic('/gazebo_muscle_interface/robot/Foot2/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateRadius1', Topic('/gazebo_muscle_interface/robot/Radius1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateRadius2', Topic('/gazebo_muscle_interface/robot/Radius2/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateHumerus1', Topic('/gazebo_muscle_interface/robot/Humerus1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateHumerus2', Topic('/gazebo_muscle_interface/robot/Humerus2/cmd_activation', Float64))
@nrp.MapRobotSubscriber('muscle_states_msg', Topic('/gazebo_muscle_interface/robot/muscle_states', MuscleStates))
@nrp.Robot2Neuron()
def transferfunction( t, clm,
                activateFoot1, activateFoot2,
                activateRadius1, activateRadius2,
                activateHumerus1, activateHumerus2,
                muscle_states_msg):
  # Obtained from Blender.
  length_bounds = {
    'Foot1': (0.004814, 0.005428),
    'Foot2': (0.004922, 0.005544),
    'Radius1': (0.005802, 0.007600),
    'Radius2': (0.004154, 0.006238),
    'Humerus1': (0.010311, 0.013530),
    'Humerus2': (0.007796, 0.011747),
  }
  if clm.value is None:
      import inspect
      import os
      clientLogger.info("Initializing Spinal Cord")
      # Configure python search paths to allow import of the spinal cord code.
      filename = inspect.getframeinfo(inspect.currentframe()).filename
      path = os.path.dirname(os.path.abspath(filename))
      path_to_code = os.path.join(path, 'resources')
      os.environ['NN_CODE'] = path_to_code
      # WARNING: Currently, code changes in imported
      # modules after initial import are ignored.
      # And that counts since the launch of the plattform.
      import simulations
      eesFreq = 0.001  # Disables external stimulation.
      eesAmp = 1 # Disables external stimulation.
      species = "mouse"
      figName  = "videoNetworkActivity"
      # The nnStructFile defines the NN configuration.
      # It is designed specifically to describe neural structures in the
      # locomotion aparatus of mice.
      nnStructFile = "closedLoopMouse.txt"
      # Instantiate the NN controller and its internal transfer functions.
      clm.value = simulations.ClosedLoopSimulation(
          nnStructFile, species , eesAmp , eesFreq, figName)
      clientLogger.info("Initializing Spinal Cord - Done")
  else:
      muscle_states =dict((m.name, m) for m in muscle_states_msg.value.muscles)
      # Prepare argument dict for input to the NN.
      mmData = {'t':t*1000+20, 'stretch':{}}
      normalized_lengths = dict(
          (k, (muscle_states[k].length - a)/(b-a)) for (k, (a,b)) in length_bounds.items()
      )
      a, b =  0.4, 0.0056
      l_CE_CF = normalized_lengths['Humerus2'] * a + b
      l_CE_PMA = normalized_lengths['Humerus1'] * a + b
      l_CE_POP = normalized_lengths['Radius2'] * a + b
      l_CE_RF  = normalized_lengths['Radius1'] * a + b
      l_CE_TA  = normalized_lengths['Foot1'] * a + b
      l_CE_LG  = normalized_lengths['Foot2'] * a + b
      mmData['stretch']['LEFT_PMA'] = l_CE_PMA
      mmData['stretch']['LEFT_CF'] = l_CE_CF
      mmData['stretch']['LEFT_POP'] = l_CE_POP
      mmData['stretch']['LEFT_RF'] = l_CE_RF
      mmData['stretch']['LEFT_LG'] = l_CE_LG
      mmData['stretch']['LEFT_TA'] = l_CE_TA
      # Advance the neural simulation.
      nnData = clm.value.run_step(mmData)
      # Activate muscles
      activateHumerus1.send_message(nnData['LEFT_PMA'])
      activateHumerus2.send_message(nnData['LEFT_CF'])
      activateRadius1.send_message(nnData['LEFT_RF'])
      activateRadius2.send_message(nnData['LEFT_POP'])
      activateFoot1.send_message(nnData['LEFT_TA'])
      activateFoot2.send_message(nnData['LEFT_LG'])

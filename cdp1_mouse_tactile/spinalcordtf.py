from std_msgs.msg import Float64
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_ros_muscle_interface.msg import MuscleStates
from gazebo_msgs.msg import ContactsState
@nrp.MapVariable('clm', initial_value=None)
@nrp.MapRobotPublisher('activateFoot1',Topic('/gazebo_muscle_interface/robot/Foot1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateFoot2',Topic('/gazebo_muscle_interface/robot/Foot2/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateRadius1', Topic('/gazebo_muscle_interface/robot/Radius1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateRadius2', Topic('/gazebo_muscle_interface/robot/Radius2/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateHumerus1', Topic('/gazebo_muscle_interface/robot/Humerus1/cmd_activation', Float64))
@nrp.MapRobotPublisher('activateHumerus2', Topic('/gazebo_muscle_interface/robot/Humerus2/cmd_activation', Float64))
@nrp.MapRobotSubscriber('muscle_states_msg', Topic('/gazebo_muscle_interface/robot/muscle_states', MuscleStates))
@nrp.MapRobotSubscriber('joint_states', Topic('/joint_states', JointState))
@nrp.MapRobotSubscriber('joint_accel', Topic('/joint_accel', Float32MultiArray))
@nrp.MapRobotSubscriber('link_states', Topic('/gazebo/link_states', LinkStates))
@nrp.MapRobotSubscriber('contacts', Topic('/gazebo/contact_point_data', ContactsState))
@nrp.Robot2Neuron()
def transferfunction( t, clm,
                activateFoot1, activateFoot2,
                activateRadius1, activateRadius2,
                activateHumerus1, activateHumerus2,
                muscle_states_msg,
                joint_states,
                joint_accel,
                link_states,
                contacts):
  from math import sqrt
  #---------------------------------------------------------
  if joint_states.value:
      # Demo section. First show joint data.
      joint_names = joint_states.value.name
      joint_pos = joint_states.value.position
      joint_effort = joint_states.value.effort
      joint_accel = joint_accel.value.data
      for n, p, e, a in zip(joint_names, joint_pos, joint_effort, joint_accel):
        clientLogger.info("%s: p=%f, e=%f, a=%f" % (n, p, e, a))
  # Then Link states. But here we just print the state of the sled.
  # The names can be obtained for example by printing out the name array here in the TF
  # or by using ROS commands - rostopic list and rostopic echo.
  if link_states.value:
      link_names = link_states.value.name
      sled_index = link_names.index('robot::cdp1_msled::sled')
      clientLogger.info("sled pose: " + str(link_states.value.pose[sled_index]))
      clientLogger.info("sled velocity: " + str(link_states.value.twist[sled_index]))
      clientLogger.info("sled accel: " + str(link_states.value.accel[sled_index]))
      # Force is the sum of all forces, including contact forces and constraint forces.
      # It is the F in F = m*a which causes the motion of the body with acceleration a and mass m.
      clientLogger.info("sled force: " + str(link_states.value.force[sled_index]))
  # Now we want to see some contact info.
  if contacts.value:
      for body_body_contacts in contacts.value.states:
          n_contact_points = len(body_body_contacts.contact_positions)
          centerx = sum([p.x for p in body_body_contacts.contact_positions])
          centery = sum([p.y for p in body_body_contacts.contact_positions])
          centerz = sum([p.z for p in body_body_contacts.contact_positions])
          center = [c/n_contact_points for c in [centerx, centery, centerz]]
          avg_normal_x = sum([p.x for p in body_body_contacts.contact_normals])
          avg_normal_y = sum([p.y for p in body_body_contacts.contact_normals])
          avg_normal_z = sum([p.z for p in body_body_contacts.contact_normals])
          l = sqrt(sum([n*n for n in [avg_normal_x, avg_normal_y, avg_normal_z]]))
          avg_normal = [n/l for n in [avg_normal_x, avg_normal_y, avg_normal_z]]
          clientLogger.info("contact " + str(body_body_contacts.collision1_name) +
                            " <-> " + str(body_body_contacts.collision2_name) + ", " +
                            str(n_contact_points) + " points:")
          clientLogger.info("  center: "+str(center))
          clientLogger.info("  avg normal: "+str(avg_normal))
          clientLogger.info("  total_wrench: "+ str(body_body_contacts.total_wrench))
  #---------------------------------------------------------
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
      import sys
      import os
      clientLogger.info("Initializing Spinal Cord")
      # Configure python search paths to allow import of the spinal cord code.
      path_to_code = os.path.join(os.environ['HBP'],'Experiments','cdp1_mouse','code')
      sys.path.insert(0, path_to_code)
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

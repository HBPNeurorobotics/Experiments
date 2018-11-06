import os
import time
import pyNN.nest as sim

import os
pathToCode = os.environ['NN_CODE']

class Simulation:
    """ Interface class to design different types of neuronal simulation.

    The simulations are based on pyNN module and

    """

    def __init__(self):
        """ Object initialization.
        """

        # variables to set in child clasess
        self.__tstop = None
        self.__integrationStep = None

        # Initialize stats
        self.simulationTime = None
        self.__printPeriod = 250 #ms

        # Setup results folder
        self._resultsFolder = pathToCode+"/../../results/"
        if not os.path.exists(self._resultsFolder):
            os.makedirs(self._resultsFolder)

    def __check_parameters(self):
        """ Check whether some parameters necessary for the simulation have been set or not. """
        if self.__tstop == None or self.__integrationStep == None:
            raise Exception("Undefined integration step and maximum time of simulation")

    def _get_tstop(self):
        """ Return the time at which we want to stop the simulation. """
        return self.__tstop

    def _set_tstop(self,tstop):
        """ Set the time at which we want to stop the simulation.

        Keyword arguments:
        tstop -- time at which we want to stop the simulation in ms.
        """
        if tstop>0: self.__tstop = tstop
        else: raise Exception("The maximum time of simulation has to be greater than 0")

    def _get_integration_step(self):
        """ Return the integration time step. """
        return self.__integrationStep

    def _set_integration_step(self,dt):
        """ Set the integration time step.

        Keyword arguments:
        dt -- integration time step in ms.
        """
        if dt>= sim.get_time_step(): self.__integrationStep = dt
        else: raise Exception("The integration step has to be greater than pyNN time step")

    def _initialize(self):
        """ Initialize the simulation. """
        self._start = time.time()
        self.__tPrintInfo=-99999

    def _integrate(self):
        """ Integrate the neuronal cells for a defined integration time step ."""
        sim.run(self.__integrationStep)

    def _update(self):
        """ Update simulation parameters. """
        raise Exception("pure virtual function")

    def _get_print_period(self):
        """ Return the period of time between printings to screen. """
        return self.__printPeriod

    def _set_print_period(self,t):
        """ Set the period of time between printings to screen.

        Keyword arguments:
        t -- period of time between printings in ms.
        """
        if t>0: self.__printPeriod = t
        else: raise Exception("The print period has to be greater than 0")

    def _print_sim_status(self):
        """ Print to screen the simulation state. """
        if sim.get_current_time()-self.__tPrintInfo>=(self.__printPeriod-0.5*self.__integrationStep):
            if self.__tPrintInfo <= 0:
                print "\nStarting the simulation:"
            self.__tPrintInfo=sim.get_current_time()
            print "\t"+str(round(sim.get_current_time()))+"ms of "+str(self.__tstop)+"ms integrated..."

    def _end_integration(self):
        """ Print the total simulation time.

        This function, executed at the end of time integration is ment to be modified
        by daughter calsses according to specific needs.
        """
        self.simulationTime = time.time()-self._start
        print "tot simulation time: "+ str(int(self.simulationTime)) + "s"

    def run(self):
        """ Run the simulation. """
        self.__check_parameters()
        self._initialize()
        while sim.get_current_time()<self.__tstop:
            self._integrate()
            self._update()
            self._print_sim_status()
        self._end_integration()

    def set_results_folder(self,resultsFolderPath):
        """ Set a new folder in which to save the results """
        self._resultsFolder = resultsFolderPath
        if not os.path.exists(self._resultsFolder):
            os.makedirs(self._resultsFolder)

    def save_results(self,name=""):
        """ Save the simulation results.

        Keyword arguments:
        name -- string to add at predefined file name (default = "").
        """
        raise Exception("pure virtual function")

    def plot(self):
        """ Plot the simulation results. """
        raise Exception("pure virtual function")

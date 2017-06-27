# ---LICENSE-BEGIN - DO NOT CHANGE OR MOVE THIS HEADER
# This file is part of the Neurorobotics Platform software
# Copyright (C) 2014,2015,2016,2017 Human Brain Project
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# ---LICENSE-END
"""
This script enables the devop to run continously in a server a list of
experiments which are specified in the experiments_list.json
"""
import json
import logging
import os
import time
from hbp_nrp_virtual_coach import virtual_coach

logger = logging.getLogger('ExperimentsLauncher')


class ExperimentsLauncher(object):
    """
    This class contains an instance of the virtual coach, and exposes a single function
    to run continuously the list of experiments contained in the experiments_list.json
    """

    def __init__(self):
        """
        In the constructor we instantiate a VC,
        and read the list of experiments from the experiments_list.json
        """
        # instantiate a virtual coach instance and an empty dictionary to
        # contain the experiments we want to run constantly
        self.__vc = virtual_coach.VirtualCoach('local')
        self.__experiments_list = {}
        #helper variables
        self.__last_status = [None]
        self.__launched = False
        self.__sim = None
        # Find where the .json with the experiments list resides. Should be
        # in the same directory as the script
        experiments_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        'experiments_list.json')
        if not os.path.isfile(experiments_path):
            raise IOError(
                'experiments_list.json not found, terminating.')
        elif not os.access(experiments_path, os.R_OK):
            raise IOError(
                'experiments_list.json is not readable, terminating.')

        # open and read the .json file
        with open(experiments_path) as experiments_file:
            try:
                self.__experiments_list = json.load(experiments_file)
            except Exception as exc:
                raise IOError(
                    'Malformed experiments_list.json: %s' % str(exc))

    def __on_status(self, status):
        """"
        Helper method that is specified as the callback function whenever
        we have a status change. By default this happens every second
        """
        self.__last_status[0] = status
        if status['state'] == 'stopped' or status['state'] == 'halted':
            self.__launched = False
            self.__sim = None

    def __wait_condition(self, timeout, condition):
        """
        Helper method that enables us to check a condition based on
        the status of the simulation. If it times out, we throw an
        exception
        """
        start = time.time()
        while time.time() < start + timeout:
            time.sleep(0.25)
            if condition(self.__last_status[0]):
                return
        raise Exception('Condition check failed')

    def run_experiments(self):
        """
        This method is running the list of experiments in an infinite loop
        """
        # we need to run the experiments continuously, thus the infinite loop
        while True:
            for experiment in self.__experiments_list:
                # necessary because sometimes the state takes a while
                # to get updated and in the meantime we cannot start
                # a new simulation
                while not self.__launched:
                    try:
                        self.__sim = self.__vc.launch_experiment(
                            str(self.__experiments_list[experiment]))
                        self.__launched = True
                    except:
                        time.sleep(1)
                # we basically update the status every second
                self.__sim.register_status_callback(self.__on_status)
                # give it some time to get a status callback
                self.__wait_condition(10, lambda x: x is not None)
                self.__sim.pause()
                time.sleep(10)
                self.__sim.start()
                self.__wait_condition(10, lambda x: x['state'] == 'started')
                while self.__launched:
                    time.sleep(1)

import subprocess
import socket
import pyNN.nest as sim
import os
from simulations import ClosedLoopSimulation


class ClosedLoopSimWebots(ClosedLoopSimulation):
    """ Simulation Manager.

    Run the musculoskeletal model and the neural network processes
    and manage the inter process communication (IPC).
    """
    def __init__(self,experiment,totSimulationTime, neuralNetworkStructure, species, eesAmplitude, eesFrequency, figName="networkActivity", dtCommunication=20):

        ClosedLoopSimulation.__init__(self, neuralNetworkStructure, species, eesAmplitude, eesFrequency, figName, dtCommunication)
        self._debug = False
        self._totSimulationTime = float(totSimulationTime)
        self._experiment = experiment

    def _run_webots(self):
        """ Run webots as a subprocess """
        if self._experiment == "hanging":
            program = ["webots","--stdout","../../sml/worlds/852_mouse_hindlimb_SCALE_tester.wbt"]
        else: raise(Exception("Unknown experiment!"))

        self._webots = subprocess.Popen(program, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Initialize TCP communication (server side)
        TCP_IP = '127.0.0.1'
        TCP_PORT = 5005
        BUFFER_SIZE = 512
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._s.bind((TCP_IP, TCP_PORT))
        self._s.listen(1)
        self._conn, addr = self._s.accept()

    def run(self):
        """ Run the closed loop simualtion. """
        self._run_webots()
        # We first wait for webots inputs
        time = sim.get_current_time()
        while time<self._totSimulationTime:
            if self._debug: print "reading data from webots:"
            mmData = self._wbt_read_data()
            if self._debug: print "running nn step..."
            nnData = self.run_step(mmData)
            self._send_data_to_wbt(nnData)
            time = sim.get_current_time()
        self.terminate()
        os.system("kill $(ps aux | grep '/Applications/Webots.app/Contents/MacOS/webots' | head -1 | awk '{print $2}')")
        self._webots.terminate()


    def _wbt_read_data(self):
        """ Read the data coming from the webots controller. """
        reaData = True
        wbtIncomingData = False
        wbtDataTemp = []
        while reaData:
            wbtIncomingMsg =  self._webots.stdout.readline().rstrip("\n").split()
            if "COMM_OUT" in wbtIncomingMsg: wbtIncomingData = True
            elif "END" in wbtIncomingMsg: reaData = False
            elif wbtIncomingData: wbtDataTemp.append(wbtIncomingMsg[1:])
            if self._debug: print "\t\t\t\tWebots: :"+" ".join(wbtIncomingMsg[1:])

        for i,data in enumerate(wbtDataTemp):
            if i==0:mmData = {'t':float(data[0]), 'stretch':{}}
            else: mmData['stretch'][data[0]] = float(data[1])
        return mmData

    def _send_data_to_wbt(self,nnData):
        """ Send the neural network's data to the webots controller. """
        nnDataString = ""
        for muscle in nnData: nnDataString+=" ".join([muscle,str(nnData[muscle])])+"\n"
        nnDataString += "END\n"
        self._conn.send(nnDataString)

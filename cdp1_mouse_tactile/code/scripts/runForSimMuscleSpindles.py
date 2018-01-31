import sys
args =  list(sys.argv)
sys.path.append('../code')
import pyNN.nest as sim
from simulations import ForSimMuscleSpindles
from SpinalNeuralNetwork import SpinalNeuralNetwork
from EES import EES
from tools import general_tools  as gt
from pyNN.utility import Timer


def main():
	""" This program launches a ForSimMuscleSpindles simulation with a NeuralNetwork structure,
	EES amplitude and EES frequency given by the user as argument. The NeuralNetwork
	need to conatain the structure of a muscle spindle sensorimotor circuitry for 2
	antagonist muscles, namely the 'TA' and 'GM'. Precomputed senosry information
	of the Ia and II fibers is used to drive the NN.
	The plots resulting from this simulation are saved in the results folder.

	This program can be executed both with and without MPI. In case MPI is used, the cells
	of the NeuralNetwork are be shared between the different hosts in order to speed up the
	simulation.

	BUG: Currently no speed is achieved with mpi...PyNN

	e.g: mpiexec -np 4 python scripts/runForSimMuscleSpindles.py 40 240 1 frwSimRat.txt test
	"""

	if len(args)<6:
		print "Error in arguments. Required arguments:"
		print "\t ees frequency [0-1000] "
		print "\t ees amplitude (0-600] or %Ia_II_Mn "
		print "\t afferent normalization factor [0-1] "
		print "\t neural network structure file .txt"
		print "\t name to add at the output files"
		print "Optional arguments:"
		print "\t time of simulation (default = -1)"
		print "\t Plot results? (1 or 0) (default = 1)"
		print "\t\t Note that in this case eesFrequency is the stimulation frequency inside the bursts."
		sys.exit(-1)

	eesFrequency = float(args[1])
	if args[2][0]=="%": eesAmplitude = [float(x) for x in args[2][1:].split("_")]
	else: eesAmplitude = float(args[2])
	norm = float(args[3])
	inputFile = args[4]
	name = args[5]
	if len(args)>=7: time = float(args[6])
	else: time = -1
	if len(args)>=8: plotResults = bool(int(args[7]))
	else: plotResults = True

	timer = Timer()
	timer.start()
	print "\nSetting up simulation environment..."
	extra = {'threads' : 1}
	sim.setup(timestep=0.1, min_delay=0.1, max_delay=2.0,**extra)
	print "\t\t\t completed in %s \n" % (timer.diff(format='long'))

	print "\nCreating simulation variables..."
	nn = SpinalNeuralNetwork(inputFile)
	ees = EES(nn,eesAmplitude,eesFrequency)
	afferentsInput = create_input(norm)
	simulation = ForSimMuscleSpindles(nn, afferentsInput, ees, None, time)
	print "\t\t\t completed in %s \n" % (timer.diff(format='long'))
	ees.get_amplitude(True)
	ees.get_frequency(True)

	simulation.run()

	if plotResults: simulation.plot("TA","GM",name,False)
	comm.Barrier()
	simulation.save_results("TA","GM",name)

	sim.end()

def create_input(norm=1):
	""" Load previously computed affarent inputs """
	afferents = {}
	afferents['TA'] = {}
	afferents['GM'] = {}
	afferents['TA']['Iaf'] = list(gt.load_txt_mpi('../afferentsFirings/meanFr_Ia_TA.txt')*norm)
	afferents['TA']['IIf'] = list(gt.load_txt_mpi('../afferentsFirings/meanFr_II_TA.txt')*norm)
	afferents['GM']['Iaf'] = list(gt.load_txt_mpi('../afferentsFirings/meanFr_Ia_GM.txt')*norm)
	afferents['GM']['IIf'] = list(gt.load_txt_mpi('../afferentsFirings/meanFr_II_GM.txt')*norm)
	dtUpdateAfferent = 5
	afferentsInput = [afferents,dtUpdateAfferent]
	return afferentsInput


if __name__ == '__main__':
	main()

import sys
args =  list(sys.argv)
import os
pathToCode = os.environ['NN_CODE']
sys.path.append(pathToCode)
from simulations import ClosedLoopSimWebots
from tools import structures_tools as tls

def main():
	""" This program executes a neural-biomechanical closed loop simulation.
	The program launches the neural netwrok (python) and the biomechanical
	model (cpp) and manages the communication between these two programs.
	The program doesn't have to be executed by MPI and doesn't run the
	neural network in a subprocess as in a the previous version.
	"""
	if len(args)<4:
		print "Error in arguments. Required arguments:"
		print "\t ees frequency [0-1000] "
		print "\t ees amplitude (0-600] or %Ia_II_Mn "
		print "\t experiment [hanging] "
		print "Optional arguments:"
		print "\t Species (mouse, rat or human)"
		print "\t tot time of simulation (default = 3000)"
		print "\t figure name (default = 'webotsNNactivity')"
		print "\t nnStructure file"

		sys.exit(-1)


	eesFreq = float(args[1])
	eesAmp = float(args[2])
	experiment = args[3]
	if len(args)>=5: species = args[4]
	else: species = "human"
	if len(args)>=6 : totTime = args[5]
	else: totTime = 3000.
	if len(args)>=7: figName  = args[6]
	else: figName  = "webotsNNactivity"

	if len(args)>=8: nnStructFile  = args[7]
	else:
		w1 = 0.045 #0.09
		w2 = 0.03 #0.03
		w3 = 0.01125 #0.045
		w4 = -0.00145 #-0.00145
		w5 = -0.0045 #-0.0045
		templateFile = "templateClosedLoop2Dof.txt"
		# templateFile = "templateClosedLoop4Dof.txt"
		# templateFile = "templateClosedLoop6Dof.txt"
		# templateFile = "templateClosedLoop.txt"
		nnStructFile = "generatedStructures/ClosedLoop_w1_%f_w2_%f_w3_%f_w4_%f_w5_%f.txt" % (w1,w2,w3,w4,w5)
		tls.modify_network_structure(templateFile,nnStructFile,None,[w1,w2,w3,w4,w5])

	clm = ClosedLoopSimWebots(experiment, totTime, nnStructFile, species, eesAmp, eesFreq , figName )
	clm.run()

if __name__ == '__main__':
	main()

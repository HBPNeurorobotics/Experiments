import sys
sys.path.append('../code')
from WebotsControl import WebotsControl

def main():
	""" This program can be used to tests a biomechanical model. """
	if len(sys.argv)<2:
		print "Error in arguments. Required arguments:"
		print "\t experiment [hanging] "
		sys.exit(-1)

	experiment = sys.argv[1]

	clm = WebotsControl(experiment)
	clm.run()

if __name__ == '__main__':
	main()

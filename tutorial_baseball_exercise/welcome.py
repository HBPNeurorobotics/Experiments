from hbp_nrp_excontrol.logs import clientLogger
@nrp.Neuron2Robot()
def welcome(t):
    if t < 30.0:
        clientLogger.advertise("""
        Welcome to the Neurorobotics Platform! 
        The guide to this tutorial is an ipython notebook located in the folder Experiments/tutorial_baseball_exercise/tutorial_baseball.ipynb
        """)

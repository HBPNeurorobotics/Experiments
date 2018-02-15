To run the Virtual Coach script cdp1_experiment.ipynb, you'll have to start a jupyter notebook with the command:

$ cle-virtual-coach jupyter notebook cdp1_experiment.ipynb

This jupyter notebook runs the cdp1 mouse experiment multiple times, each time with a different neural network structure file. It loops through all the generated structures found in cdp1_mouse/nn_structures/generatedStructures.

Each neural network structure is simulated for a predefined simulation time that can be set in the third code cell in the notebook. The default value is 10 seconds for each run. Make sure to keep a copy of cdp1_mouse/nn_structures/closedLoopMouse.txt as it will be overwritten by the generated structures.

After each run, the sled positions during the simulation are saved in a list to be plotted in the last cell of the notebook.

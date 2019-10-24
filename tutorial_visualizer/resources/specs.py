###############################
### Some useful definitions ###
###############################


# Imports
import numpy as np
import nest
import re
from pyNN.connectors import Connector


# Create a custom connector (use sim.Projection explicitly to go faster)
class MyConnector(Connector):
    def __init__(self, source, target):
        self.source = source
        self.target = target
    def connect(self, projection):
        nest.Connect([projection.pre.all_cells[s] for s in self.source], [projection.post.all_cells[t] for t in self.target], 'one_to_one', syn_spec=projection.nest_synapse_model)


# Create different filters for orientation selectivity
def createConvFilters(nOri, size, phi, sigmaX, sigmaY, oLambda):

    # Initialize the filters
    filters = np.zeros((nOri, size, size))

    # Fill them with gabors
    midSize = (size-1.)/2.0
    maxValue = -1
    for k in range(0, nOri):
        theta = np.pi*(k+1)/nOri + phi
        for i in range(0, size):
            for j in range(0, size):
                x = (i-midSize)*np.cos(theta) + (j-midSize)*np.sin(theta)
                y = -(i-midSize)*np.sin(theta) + (j-midSize)*np.cos(theta)
                filters[k][i][j] = np.exp(-((x*x)/sigmaX + (y*y)/sigmaY)) * np.sin(2*np.pi*x/oLambda)

    # Normalize the orientation filters so that they have the same overall strength
    for k in range(nOri):
        filters[k] /= np.sqrt(np.sum(filters[k]**2))

    # Return the filters to the computer
    return (filters, -filters)


# Write a custom configuration file for the brain visualizer
def writeJson(path='visualizer_tutorial.py'):

    # Parameters initialization
    currentLine = 0
    cellType    = None
    imRows      = 0
    imCols      = 0
    nOri        = 0

    # Run through the brain file to build the ID of all the populations
    popID          = []
    nNeuronTot     = 0
    descriptorKeys = ['featList', 'imRows', 'imCols']
    with open(path, 'r') as inputFile:
        for line in inputFile:

            # Look for parameters
            if 'cellType'   in line and cellType   == None:
                cellType    = re.sub('\s+', '', line.split(".")[1].split("(")[0])
            if 'nOri'       in line and nOri       == 0:
                nOri        = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
            if 'imRows'     in line and imRows      == 0:
                imRows      = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
                oriRows     = imRows+1
            if 'imCols'     in line and imCols      == 0:
                imCols      = int(re.sub('\s+', '', line.split("=")[1].split("#")[0]))
                oriCols     = imCols+1
                spaceOri    = int(imCols/2)

            # Look for the neuron populations
            if 'sim.Population(' in line:

                # Find the name and the type of the population
                name      =      re.sub('\s+', '', line.split('=')[0])
                typeValue = eval(re.sub('\s+', '', line.split(',')[-1])[:-1])

                # Isolate the descriptors from the nest.Create(...) lines
                descriptorValues = line.split('*')                                                          # descriptors are separated by asterisks
                descriptorValues[ 0] = re.sub('\s+', '', descriptorValues[0].split("(")[-1].split(")")[0])  # remove what's behind the first feature
                descriptorValues[-1] = re.sub('\s+', '', descriptorValues[-1].split(",")[0])                # remove what's after  the last  feature

                # Convert the string-like descriptor values into real values
                descriptorValues = [eval(valString) for valString in descriptorValues]

                # Find the number of neurons in this pop and add it to the total value
                nNeuronPop  = np.prod(descriptorValues)
                nNeuronTot += nNeuronPop

                # Take care if several features exist in the population (orientation, flow, etc.), i.e. when there is more than 4 descriptors
                descriptorValues = [list(descriptorValues[0:-2]), descriptorValues[-2], descriptorValues[-1]]

                # Create the descriptor dictionary
                descriptor = {'type': typeValue, 'line': currentLine}
                currentLine += 1
                for index, descriptorKey in enumerate(descriptorKeys):
                    descriptor[descriptorKey] = descriptorValues[index]

                # Insert the population as an entry of the popID dictionary
                popID.append((name, descriptor))

    # Open and write the file containing the neuron locations
    with open("neuronPositions.json", "w") as outputFile:
        outputFile.write('{"populations": {')
        nPopulations = len(popID)
        xDraw = -15*int(nPopulations/2)    # x-coordinate of the current set of (pop, feature) ; is incremented throughout the loop
        for popIdx in range(nPopulations):

            # Control the population is made of neurons
            popName = popID[popIdx][0]
            thisPop = popID[popIdx][1]
            if thisPop['type'] == 'IF_curr_alpha':

                # Starts writing for this population
                outputFile.write('\n\t"%s": {\n\t\t"neurons": {' % popName)
                neuronIndex = 0

                # Start to draw the neurons : x = pops, y = feature, (z, y) = (row, col)
                totalNumFeatures = np.prod(thisPop['featList'])
                for f in range(totalNumFeatures):

                    # Compute y and z coordinate origins
                    yTopLeft = (thisPop['imCols']+spaceOri)*int((f+1)/2 + 0.4)*np.power(-1, f+1) - int(thisPop['imCols']/2)
                    if totalNumFeatures%2 == 0:
                        yTopLeft = yTopLeft - int((thisPop['imCols']+spaceOri)/2)
                    zTopLeft = 0 - int(thisPop['imRows']/2)

                    # Loop throughout all the neurons of this feature inside this population
                    for zRow in range(thisPop['imRows']):
                        for yCol in range(thisPop['imCols']):

                            # Write the (x,y,z)-coordinates of this neuron
                            yDraw = yTopLeft + yCol
                            zDraw = zTopLeft + zRow
                            outputFile.write('\n\t\t\t"%s": {"pos": [%s, %s, %s]}' % (neuronIndex, xDraw, zDraw, yDraw))
                            neuronIndex += 1
                            if (f+1)*(zRow+1)*(yCol+1) < np.prod(thisPop['featList'])*thisPop['imRows']*thisPop['imCols']:
                                outputFile.write(", ")

                xDraw += 25  # space on the x-axis between each population

                # Write the end of the line
                if popIdx < nPopulations-1:
                    outputFile.write('}},')
                else:
                    outputFile.write('}}\n}}')

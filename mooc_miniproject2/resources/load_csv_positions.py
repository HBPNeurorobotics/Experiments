
import numpy
import csv
import math

def load_lattice(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]

    # Re-create the SOM lattice
    Nn = int(math.sqrt(len(data)/2))
    lattice = numpy.zeros((Nn,Nn,2))
    for i, line in enumerate(data):
        z = line[0]; y = line[1]; x = line[2]
        lattice[int(x), int(y), int(z)] = line[3]
    return lattice

def load_waypoints(filepath):
    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        data = [r for r in reader]

    # Re-create the waypoints
    Nw = int(len(data))
    waypoints = numpy.zeros((Nw,2))
    for i, line in enumerate(data):
        waypoints[i, 0] = line[0]
        waypoints[i, 1] = line[1]
    return waypoints
        
    
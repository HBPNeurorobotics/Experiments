### Simulation parameters ###

retina.TempStep(int(1000/30)) # simulation step (in ms)
retina.SimTime(1200) # simulation time (in ms)
retina.NumTrials(1) # number of trials
retina.PixelsPerDegree(5.0) # pixels per degree of visual angle
retina.NRepetitions(100) # number of simulation steps every image in the input sequence is repeated
retina.DisplayDelay(0) # display delay
retina.DisplayZoom(1.0) # display zoom
retina.DisplayWindows(3) # Displays per row

### Visual input ###

# Folder that contains the input sequence
# retina.Input('sequence', {'path': '../input_sequences/Weberlaw/0_255/'})
retina.Input('camera', {'size': (320, 240)})

### Creation of computational retinal microcircuits ###

# Temporal modules
retina.Create('LinearFilter', 'tmp_photoreceptors_S', {
    'type': 'Gamma',
    'tau': 30.0,
    'n': 10.0
})
retina.Create('LinearFilter', 'tmp_photoreceptors_M', {
    'type': 'Gamma',
    'tau': 30.0,
    'n': 10.0
})
retina.Create('LinearFilter', 'tmp_photoreceptors_L', {
    'type': 'Gamma',
    'tau': 30.0,
    'n': 10.0
})
retina.Create('LinearFilter', 'tmp_horizontal_OFF', {
    'type': 'Gamma',
    'tau': 20.0,
    'n': 1.0
})
retina.Create('SingleCompartment', 'tmp_bipolar_OFF', {
    'number_current_ports': 1.0,
    'number_conductance_ports': 1.0,
    'Rm': 1.0,
    'tau': 10.0,
    'Cm': 100.0,
    'E': 0.0
})
retina.Create('LinearFilter', 'tmp_amacrine_OFF', {
    'type': 'Gamma',
    'tau': 10.0,
    'n': 1.0
})

# Additional layer
retina.Create('LinearFilter', 'tmp_horizontal_ON', {
    'type': 'Gamma',
    'tau': 20.0,
    'n': 1.0
})
retina.Create('SingleCompartment', 'tmp_bipolar_ON', {
    'number_current_ports': 1.0,
    'number_conductance_ports': 1.0,
    'Rm': 1.0,
    'tau': 10.0,
    'Cm': 100.0,
    'E': 0.0
})
retina.Create('LinearFilter', 'tmp_amacrine_ON', {
    'type': 'Gamma',
    'tau': 10.0,
    'n': 1.0
})

# Spatial filters
retina.Create('GaussFilter', 'Gauss_horizontal_OFF', {
    'sigma': 0.3,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_bipolar_OFF', {
    'sigma': 0.1,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_amacrine_OFF', {
    'sigma': 0.3,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_ganglion_OFF', {
    'sigma': 0.2,
    'spaceVariantSigma': False
})

# Spatial filters (additional layer)
retina.Create('GaussFilter', 'Gauss_horizontal_ON', {
    'sigma': 0.3,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_bipolar_ON', {
    'sigma': 0.1,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_amacrine_ON', {
    'sigma': 0.3,
    'spaceVariantSigma': False
})
retina.Create('GaussFilter', 'Gauss_ganglion_ON', {
    'sigma': 0.2,
    'spaceVariantSigma': False
})

# Common nonlinearities
retina.Create('StaticNonLinearity', 'SNL_photoreceptors_L', {
    'slope': -0.1,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_photoreceptors_M', {
    'slope': -0.1,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_photoreceptors_S', {
    'slope': -0.1,
    'offset': 0.0,
    'exponent': 1.0
})

# Nonlinearities
retina.Create('StaticNonLinearity', 'SNL_photoreceptors_OFF', {
    'slope': -0.1,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_horizontal_OFF', {
    'slope': 1.0,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_amacrine_OFF', {
    'slope': 0.2,
    'offset': 1.0,
    'exponent': 2.0
})
retina.Create('StaticNonLinearity', 'SNL_bipolar_OFF', {
    'slope': 10.0,
    'offset': 0.0,
    'exponent': 1.0,
    'threshold': 0.0
})
retina.Create('StaticNonLinearity', 'SNL_ganglion_OFF', {
    'slope': 5.0,
    'offset': 0.0,
    'exponent': 1.0
})

# Additional layer for off centre
retina.Create('StaticNonLinearity', 'SNL_photoreceptors_ON', {
    'slope': -0.1,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_horizontal_ON', {
    'slope': 1.0,
    'offset': 0.0,
    'exponent': 1.0
})
retina.Create('StaticNonLinearity', 'SNL_amacrine_ON', {
    'slope': 0.2,
    'offset': 1.0,
    'exponent': 2.0
})
retina.Create('StaticNonLinearity', 'SNL_bipolar_ON', {
    'slope': 10.0,
    'offset': 0.0,
    'exponent': 1.0,
    'threshold': 0.0
})
retina.Create('StaticNonLinearity', 'SNL_ganglion_ON', {
    'slope': 5.0,
    'offset': 0.0,
    'exponent': 1.0
})


### Connections ###

# Phototransduction
retina.Connect('S_cones', 'tmp_photoreceptors_S', 'Current')
retina.Connect('M_cones', 'tmp_photoreceptors_M', 'Current')
retina.Connect('L_cones', 'tmp_photoreceptors_L', 'Current')

# common connections
retina.Connect('tmp_photoreceptors_S', 'SNL_photoreceptors_S', 'Current')
retina.Connect('tmp_photoreceptors_M', 'SNL_photoreceptors_M', 'Current')
retina.Connect('tmp_photoreceptors_L', 'SNL_photoreceptors_L', 'Current')

retina.Connect(['tmp_photoreceptors_L', '-', 'tmp_photoreceptors_M'], 'SNL_photoreceptors_OFF', 'Current')
retina.Connect(['tmp_photoreceptors_M', '-', 'tmp_photoreceptors_L'], 'SNL_photoreceptors_ON', 'Current')

# Horizontal cells
retina.Connect('SNL_photoreceptors_OFF', 'Gauss_horizontal_OFF', 'Current')
retina.Connect('Gauss_horizontal_OFF', 'tmp_horizontal_OFF', 'Current')
retina.Connect('tmp_horizontal_OFF', 'SNL_horizontal_OFF', 'Current')

# Subtraction at Outer Plexiform Layer
retina.Connect(['SNL_horizontal_OFF', '-', 'SNL_photoreceptors_OFF'], 'Gauss_bipolar_OFF', 'Current')
retina.Connect('Gauss_bipolar_OFF', 'tmp_bipolar_OFF', 'Current')
retina.Connect('tmp_bipolar_OFF', 'SNL_bipolar_OFF', 'Current')

# Gain control at Inner Plexiform Layer
retina.Connect('SNL_bipolar_OFF', 'Gauss_amacrine_OFF', 'Current')
retina.Connect('Gauss_amacrine_OFF', 'tmp_amacrine_OFF', 'Current')
retina.Connect('tmp_amacrine_OFF', 'SNL_amacrine_OFF', 'Current')
retina.Connect('SNL_amacrine_OFF', 'tmp_bipolar_OFF', 'Conductance')

# Bipolar-ganglion synapse
retina.Connect('SNL_bipolar_OFF', 'Gauss_ganglion_OFF', 'Current')
retina.Connect('Gauss_ganglion_OFF', 'SNL_ganglion_OFF', 'Current')

#######################
# Horizontal cells
retina.Connect('SNL_photoreceptors_ON', 'Gauss_horizontal_ON', 'Current')
retina.Connect('Gauss_horizontal_ON', 'tmp_horizontal_ON', 'Current')
retina.Connect('tmp_horizontal_ON', 'SNL_horizontal_ON', 'Current')

# Subtraction at Outer Plexiform Layer
retina.Connect(['SNL_horizontal_ON', '-', 'SNL_photoreceptors_ON'], 'Gauss_bipolar_ON', 'Current')
retina.Connect('Gauss_bipolar_ON', 'tmp_bipolar_ON', 'Current')
retina.Connect('tmp_bipolar_ON', 'SNL_bipolar_ON', 'Current')

# Gain control at Inner Plexiform Layer
retina.Connect('SNL_bipolar_ON', 'Gauss_amacrine_ON', 'Current')
retina.Connect('Gauss_amacrine_ON', 'tmp_amacrine_ON', 'Current')
retina.Connect('tmp_amacrine_ON', 'SNL_amacrine_ON', 'Current')
retina.Connect('SNL_amacrine_ON', 'tmp_bipolar_ON', 'Conductance')

# Bipolar-ganglion synapse
retina.Connect('SNL_bipolar_ON', 'Gauss_ganglion_ON', 'Current')
retina.Connect('Gauss_ganglion_ON', 'SNL_ganglion_ON', 'Current')

# Connection with NEST
retina.Connect(['SNL_ganglion', '+', 'SNL_ganglion_ON'], 'Output', 'Current')

### Displays and data analysis  ###

retina.Show('Input', True, {'margin': 0})
# retina.Show('tmp_photoreceptors', True, {'margin': 0}) # L cones
retina.Show('tmp_photoreceptors_ON', True, {'margin': 0}) # M cones
# retina.Show('SNL_photoreceptors_OFF', True, {'margin': 0})
retina.Show('SNL_photoreceptors_ON', True, {'margin': 0})
# retina.Show('SNL_horizontal', True, {'margin': 0})
# retina.Show('SNL_bipolar', True, {'margin': 0})
# retina.Show('SNL_amacrine', True, {'margin': 0})
retina.Show('SNL_ganglion_OFF', True, {'margin': 0})
retina.Show('SNL_ganglion_ON', True, {'margin': 0})

# Spatial multimeters of row/col 12th at 200 ms
# row selection
"""
retina.Multimeter('spatial', 'Horizontal cells', 'SNL_horizontal', {
    'timeStep': 200,
    'rowcol': True,
    'value': 12
    }, True)
# col selection
retina.Multimeter('spatial', 'Horizontal cells', 'SNL_horizontal', {
    'timeStep': 200,
    'rowcol': False,
    'value': 12
    }, True)

# Temporal multimeter of ganglion cell at (5,5)
retina.Multimeter('temporal', 'Ganglion cell', 'SNL_ganglion', {'x': 5, 'y': 5}, True)
"""

@nrp.MapRobotSubscriber("laser", Topic('/p3dx_green/laser/scan', sensor_msgs.msg.LaserScan))
    @nrp.MapSpikeSource("green_laser_on", nrp.map_neurons(range(0, 45), lambda i: nrp.brain.green_laser_on[i]), nrp.dc_source)
    @nrp.MapSpikeSource("green_laser_off", nrp.map_neurons(range(0, 45), lambda i: nrp.brain.green_laser_off[i]), nrp.dc_source)
    @nrp.Robot2Neuron()
    def read_green_laser(t, laser, green_laser_off, green_laser_on):
        if laser.value is not None:
            weights = [x != float('inf') and x < 0.5 for x in laser.value.ranges]
            for i in xrange(45):
                green_laser_on[i].amplitude = 0.5 * weights[8*i:8*(i+1)].count(True)
                green_laser_off[i].amplitude = 0.5 * weights[8*i:8*(i+1)].count(False)

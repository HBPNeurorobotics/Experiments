@nrp.MapRobotSubscriber("laser", Topic('/p3dx_red/laser/scan', sensor_msgs.msg.LaserScan))
    @nrp.MapSpikeSource("red_laser_on", nrp.map_neurons(range(0, 45), lambda i: nrp.brain.red_laser_on[i]), nrp.dc_source)
    @nrp.MapSpikeSource("red_laser_off", nrp.map_neurons(range(0, 45), lambda i: nrp.brain.red_laser_off[i]), nrp.dc_source)
    @nrp.Robot2Neuron()
    def read_red_laser(t, laser, red_laser_on, red_laser_off):
        if laser.value is not None:
            weights = [x != float('inf') and x < 0.5 for x in laser.value.ranges]
            for i in xrange(45):
                red_laser_on[i].amplitude = 0.5 * weights[4*i:4*(i+1)].count(True)
                red_laser_off[i].amplitude = 0.5 * weights[4*i:4*(i+1)].count(False)
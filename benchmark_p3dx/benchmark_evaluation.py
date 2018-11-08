# Imported Python Transfer Function
@nrp.MapRobotPublisher('benchmark_metrics', Topic('/benchmark/metrics', sensor_msgs.msg.Image))
@nrp.MapVariable("metric", global_key="metric", initial_value=None)
# 'robot_index' is the index of the values related to the Pioneer 3DX robot in the 'position.value.pose' data structure
@nrp.MapVariable("robot_index", global_key="robot_index", initial_value=None)
@nrp.MapVariable("is_over", global_key="is_over", initial_value=False)
@nrp.MapRobotSubscriber("position", Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.Robot2Neuron()
def benchmark_evaluation(t, metric, robot_index, position, is_over, benchmark_metrics):
    from math import sqrt
    from math import exp
    from math import acos

    class SquarePathMetric(object):
        """Class used to handle the metric of the square path benchmark."""

        class PathSegment(object):
            """
            Class used to represent one side of the square.

            It handles the performance of the robot for this segment.
            """

            # Map between pairs of quarters and the vertex in between.
            # This is used to detect when the robot has reached the
            # next segment.
            QUARTERS2VERTEX = [
                [None, 1, 0, 0],
                [1, None, 2, 1],
                [2, 2, None, 3],
                [0, 3, 3, None]
            ]

            initialized = False

            def getQuarter(self, point):
                """
                Given a 2d point, returns the corresponding quarter ID.

                The path of the robot is split in 4 segments, each segment corresponds
                to one edge of the square. Since the robot is not exactly on the edge, we
                need a way to map any point to one of the 4 edges. To do this we split the
                world into 4 quarters using the extended diagonals of the square. Each
                quarter corresponds to one edge of the square. The ID of each quarter
                corresponds to the order in which the robot will go through them.
                side: size of one side of the square, in meters.
                point: tuple containing x and y coordinates.
                returns 0, 1, 2 or 3 depending on the quarter in which the point is located
                """
                # Checks on which side of the bottom-left to top-right diagonal the point
                # is.
                posDiag = point[0] + point[1] > 0

                # Checks on which side of the top-left to bottom-right diagonal the point
                # is.
                negDiag = point[1] > point[0]

                if posDiag:
                    if negDiag:
                        return 0
                    else:
                        return 1
                if negDiag:
                    return 3
                return 2

            def isInitialized(self):
                return self.initialized

            def init(self, length, goal, goalID, orientation):
                """
                Default constructor.

                length: length of the segment, in meters.
                goal: tupple containing the 2d coordinates of the goal for this
                segment.
                goalID: ID of the vertex corresponding to the end of the segment
                orientation: 0 or 1 if the path is parallel to the X axis or
                the Y axis, respectively.
                """
                self.initialized = True
                self.length = length
                self.goal = goal
                self.goalID = goalID
                self.orientation = orientation

                self.minPath = goal[orientation]
                self.maxPath = goal[orientation]
                self.currentPoint = None
                self.currentAngle = None
                self.currentTime = None
                self.startTime = None

                self.totalTime = None

                self.performance = 0
                self.goalReached = False

                self.performanceInit = False

            def setPerformanceParam(self, linearPart, maxTime,
                                    distanceWeight, pathWeight, timeWeight):
                """
                Set the parameters used to compute the performance for this segment.

                This method must be called at least once or the performance will not
                be computed.

                linearPart: weight of the linear component of the distance part of
                the performance.
                maxTime: maximum time before the time part of the performance is 0, in
                seconds.
                distanceWeight: weight of the distance part in the performance.
                pathWeight: weight of the path part in the performance.
                timeWeight: weight of the time part in the performance.
                """
                self.linearPart = linearPart
                self.expPart = 1 - linearPart

                self.maxTime = maxTime

                self.distanceWeight = distanceWeight
                self.pathWeight = pathWeight
                self.timeWeight = timeWeight
                self.totalWeight = distanceWeight + pathWeight + timeWeight

                self.performanceInit = True

            def getPerformance(self):
                """Return the current performance for this segment."""
                return self.performance

            def isGoalReached(self):
                """
                Check if robot has reached the goal.

                Return True if the robot has already reached the goal,
                or False otherwise.
                """
                return self.goalReached

            def timeStopped(self, currentTime):
                """
                Return how much time since the robot has stopped.

                Returns the difference between the time supplied in argument, and the
                last time the position of the robot was updated, or 0 if the robot
                position was never updated.

                currentTime: current time in the simulation.
                """
                if self.currentTime is None:
                    return 0
                return currentTime - self.currentTime

            def update(self, point, angle, time):
                """
                Update the position of the robot.

                Updates the position of the robot on this segment and computes a new
                performance accordingly.

                point: tupple with the coordinates of the robot.
                angle: orientation of the robot, in radian.
                time: current time of the simulation, in seconds.
                """
                # If the point is outside of the current corridor, enlarge the corridor
                # to include the point.
                if point[self.orientation] < self.minPath:
                    self.minPath = point[self.orientation]
                elif point[self.orientation] > self.maxPath:
                    self.maxPath = point[self.orientation]

                if (self.startTime is None):
                    self.startTime = time
                    self.currentTime = time


                # Reduce precision of point and angle
                point[0] = round(point[0], 3)
                point[1] = round(point[1], 3)
                angle = round(angle, 3)

                # If the robot position or orientation has changed, we need to update
                # the performance.
                if self.currentPoint != point or self.currentAngle != angle:
                    # Checks if the robot has crossed a diagonal (=reached the goal).
                    if self.currentPoint is not None:
                        quarterBefore = self.getQuarter(self.currentPoint)
                        quarterAfter = self.getQuarter(point)
                        if self.QUARTERS2VERTEX[quarterBefore][quarterAfter] == self.goalID:
                            self.goalReached = True

                    # Updates the position of the robot.
                    self.currentPoint = point
                    self.currentTime = time
                    self.currentAngle = angle

                    if (self.performanceInit):
                        # Computes distance performance.
                        distX = self.goal[0] - self.currentPoint[0]
                        distY = self.goal[1] - self.currentPoint[1]
                        distance = (sqrt(distX**2 + distY**2) / self.length)
                        distanceExp = exp(-6 * distance)
                        distancePerformance = ((self.linearPart * -distance) +
                                               (self.expPart * distanceExp))

                        # Computes path performance.
                        pathPerformance = 1 - min((self.maxPath - self.minPath) / self.length, 1)

                        # Computes time performance.
                        self.totalTime = self.currentTime - self.startTime
                        timePerformance = max(1 - (self.totalTime / self.maxTime), 0)

                        # Computes average for this segment.
                        self.performance = (distancePerformance * self.distanceWeight +
                                            pathPerformance * self.pathWeight +
                                            timePerformance * self.timeWeight) / self.totalWeight


        # Amount of time, in seconds, the robot is allowed to stop before the
        # supervisor considers the robot has finished.
        MAX_STOP_TIME = 2

        # The maximum time, in seconds, for the robot to reach a goal.
        # This is used to compute the time performance of a segment (if a robot takes more
        # than that, the performance will be 0) and the total time of the simulation (the
        # supervisor will end after four times this value has passed).
        MAX_TIME = 20

        # The maximum time, in seconds, for the robot to complete all the segments.
        # By default this is based on the value of MAX_TIME.
        MAX_TOTAL_TIME = 4 * MAX_TIME

        # Weight of the different parts of the performance.
        # The overall performance is a weighted average of the different parts.
        TIME_WEIGHT = 0.5
        DISTANCE_WEIGHT = 1
        PATH_WEIGHT = 1

        # The distance metric uses 2 component, one is linear and one is exponential.
        # This is the weight of the linear component.
        # The weight of the exponential component will be 1 - LINEAR_PART.
        LINEAR_PART = 0.1


        # Ratio between the coordinate of the world and the web interface
        # canvas
        WEB_INTERFACE_SCALE = 50

        initialized = False

        def isInitialized(self):
            return self.initialized

        def init(self, storePoints):
            """
            Default constructor.

            storePoints: when True, points added with update will be stored in
            a buffer to be sent to the web interface canvas with the
            getWebNewPoints() method.
            """
            self.initialized = True
            self.rendered_image = None
            self.currentSegment = 0
            self.robotHasFinished = False
            self.storePoints = storePoints
            self.newPoints = []

            # Size of one edge of the square, in meters.
            SQUARE_EDGE_SIZE = 2

            # Creates 4 instances of PathSegment for the 4 segments.
            self.segments = [
                self.PathSegment(),
                self.PathSegment(),
                self.PathSegment(),
                self.PathSegment()
            ]
            self.segments[0].init(SQUARE_EDGE_SIZE, (SQUARE_EDGE_SIZE, 0), 1, 1)
            self.segments[1].init(SQUARE_EDGE_SIZE, (SQUARE_EDGE_SIZE, SQUARE_EDGE_SIZE), 2, 0)
            self.segments[2].init(SQUARE_EDGE_SIZE, (0, SQUARE_EDGE_SIZE), 3, 1)
            self.segments[3].init(SQUARE_EDGE_SIZE, (0, 0), 0, 0)

            for i in range(0, 4):
                self.segments[i].setPerformanceParam(self.LINEAR_PART,
                                                     self.MAX_TIME,
                                                     self.DISTANCE_WEIGHT,
                                                     self.PATH_WEIGHT,
                                                     self.TIME_WEIGHT)

        def update(self, point, angle, time):
            """
            Update the metric given the current position of the robot.

            point: tuple containing x and z coordinates of the robot.
            angle: orientation, in radian, of the robot.
            time: seconds elapsed since the beginning of the benchmark.
            """

            if time < self.MAX_TOTAL_TIME:

                if self.storePoints:
                    self.newPoints.append(point)

                # Updates the position of the robot in the current segment.
                self.segments[self.currentSegment].update(point, angle, time)

                # Checks if robot has stopped.
                if (self.segments[self.currentSegment].timeStopped(time) >
                   self.MAX_STOP_TIME):
                    self.robotHasFinished = True
                    clientLogger.advertise(
                        "Benchmark is completed.\nSegment 1: %.4f\nSegment 2: %.4f\nSegment 3: %.4f\nSegment 4: %.4f\nPerformance: %.4f\n" \
                        % (self.getSegmentPerformance(0), self.getSegmentPerformance(1),
                           self.getSegmentPerformance(2), self.getSegmentPerformance(3), self.getPerformance()))

                # Checks if robot has reached the goal of the current segment.
                # If yes, switch to the next segment (unless we are in the last
                # one).
                if (self.segments[self.currentSegment].isGoalReached() and
                   self.currentSegment < 3):
                    self.currentSegment = self.currentSegment + 1
            else:
                self.robotHasFinished = True
                clientLogger.info("TIMEOUT")
                clientLogger.advertise(
                    "Benchmark is over because of timeout.\nSegment 1: %.4f\nSegment 2: %.4f\nSegment 3: %.4f\nSegment 4: %.4f\nPerformance: %.4f\n" \
                     % (self.getSegmentPerformance(0), self.getSegmentPerformance(1),
                        self.getSegmentPerformance(2), self.getSegmentPerformance(3), self.getPerformance()))

        def isBenchmarkOver(self):
            """
            Check if the benchmark is over.

            Return true if the benchmark is over, or false otherwise.
            """
            return self.robotHasFinished

        def getSegmentPerformance(self, segment):
            """
            Return the performance for a segment.

            segment: index of the segment
            """
            if (segment < 0 or segment > 3):
                return 0

            return self.segments[segment].getPerformance()

        def getPerformance(self):
            """Return the overall performance."""
            performance = 0

            # For each segment, adds the grade to the total, and displays
            # corresponding label.
            for i in range(0, 4):
                performance = performance + self.segments[i].getPerformance()

            # Computes average.
            performance = performance / 4

            return performance

        def getImageWithMetrics(self):
            """
            Return image showing metrics and robot trajectory
            """
            import numpy as np
            import PIL
            from PIL import ImageFont
            from PIL import Image
            from PIL import ImageDraw
            offsetX = 20
            offsetY = 85
            if self.rendered_image is None:
                self.rendered_img = Image.new('RGB', (256, 256), "white")
                draw = ImageDraw.Draw(self.rendered_img)
                regularFont = ImageFont.truetype("LiberationMono-Regular.ttf", 16)
                boldFont = ImageFont.truetype("LiberationMono-Bold.ttf", 16)
                # Write metrics text
                performance = 0
                for i in range(1, 5):
                    performance = performance + self.segments[i-1].getPerformance()
                    text = ('Segment %d: %5.2f%%') % (i, self.segments[i-1].getPerformance() * 100)
                    draw.text((10, i * 20 - 15), text, fill="black", font=regularFont)
                text = ('Average:   %5.2f%%') % (performance * 25)
                draw.text((10, 90), text, fill="black", font=boldFont)
                # Draw square and segment ids in gray
                grayColor = (187, 187, 187)
                draw.rectangle([(50 + offsetX, 50 + offsetY), (150 + offsetX, 150 + offsetY)], outline=grayColor)
                font = ImageFont.truetype("LiberationSerif-Regular.ttf", 18)
                draw.text((35  + offsetX, 90 + offsetY), '4', fill=grayColor, font=regularFont)
                draw.text((97  + offsetX, 30  + offsetY), '1', fill=grayColor, font=regularFont)
                draw.text((155 + offsetX, 90 + offsetY), '2', fill=grayColor, font=regularFont)
                draw.text((97  + offsetX, 155 + offsetY), '3', fill=grayColor, font=regularFont)

            # Draw trajectory in red
            draw = ImageDraw.Draw(self.rendered_img)
            for point in self.newPoints:
                x = (int(point[0] * 50)) + offsetX + 100
                y = offsetY + 100 - (int(point[1] * 50))
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 0, 0))
            cv_img = np.array(self.rendered_img)
            msg_frame = CvBridge().cv2_to_imgmsg(cv_img, 'rgb8')
            return msg_frame

    if metric.value == None:
        clientLogger.info(
              "The metric used to evaluate the robot is applied for 4 separate segments of the path, " \
            + "which correspond to the 4 sides of the square.\n" \
            + "Each segment is defined as a corridor that lies on one edge of the square. " \
            + "The \"goal\" of one segment is defined as the vertex between the current and the next segment. " \
            + "To reach the next segment, the robot must cross the line going through the center of the square and the 'goal' vertex.\n" \
            + "For each individual segment, we compute a performance which is based on 3 different parameters: " \
            + "the 'path' (how well the robot managed to keep close to the \"ideal\" route), " \
            + "the 'time' needed to go through this segment, and the 'distance' to the goal, " \
            + "which is mostly used to evaluate how close to the goal the robot is in the current segment.\n"
            + "The 'path' element is a linear value inversely proportional to the breadth of the corridor containing the path of the robot for this segment. " \
            + "It goes from 1 when the robot went on a perfect line, to 0 when the corridor is wider than it is long.\n" \
            + "The 'time' element is a linear value inversely proportional to the time needed to go through this segment. " \
            + "It goes from 1 for a duration of 0 seconds to 0 for 20 seconds or more.\n " \
            + "The 'distance' element is how far to the goal (one vertex of the square) is the robot in the current segment. " \
            + "If the robot has already completed this segment, this will be how far was the robot when he exited this segment. "\
            + "This value goes from 1 for a distance of 0 to 0 when the distance is larger than one side of the square. " \
            + "The value is defined as: 0.1 * (1 - distance/2) + 0.9 * exp(-3 * distance). " \
            + "The exponential part is meant to reward a robot which is very precise when reaching the goal, while the linear part " \
            + "is there so a robot which is moving in the right direction isn't losing points due to the time portion of the performance. "\
            + "Only a slow robot should lose points, whereas an average robot will simply gain less than a fast one.\n" \
            + "The performance for the segment is a weighted average of the 3 elements. " \
            + "The time element has half the weight of the other two.\n" \
            + "The overall performance is the average of the 4 segments."
        )
        metric.value = SquarePathMetric()
        metric.value.init(True)

    if position.value != None:
        if not metric.value.isBenchmarkOver():
            # determine if previously set robot index has changed
            if robot_index.value is not None:
                # if the value is invalid, reset the index below
                if robot_index.value >= len(position.value.name) or\
                    position.value.name[robot_index.value] != 'p3dx':
                    robot_index.value = None
            # robot index is invalid, find and set it
            if robot_index.value is None:
                # 'p3dx' is the bodyModel declared in the bibi, if not found raise error
                robot_index.value = position.value.name.index('p3dx')
            # Get current time and position/orientation of the robot.
            pos = position.value.pose[robot_index.value].position
            pos2d = [pos.x, pos.y]
            orientation = position.value.pose[robot_index.value].orientation
            angle = 2 * acos(orientation.w)
            metric.value.update(pos2d, 2 * acos(orientation.w), t)
            benchmark_metrics.send_message(metric.value.getImageWithMetrics())

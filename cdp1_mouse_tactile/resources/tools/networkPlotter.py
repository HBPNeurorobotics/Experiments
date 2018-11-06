import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.animation as animation
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Neuron():
    def __init__(self, x, y, radius=0.25,linewidth=2.5):
        self._x = x
        self._y = y
        self._radius = radius
        self._linewidth = linewidth
        self.defaultColor = "#888888"

    def draw(self,ax):
        self._circle = plt.Circle((self._x, self._y), radius=self._radius, edgecolor=self.defaultColor, facecolor=self.defaultColor, linewidth=self._linewidth)
        ax.add_patch(self._circle)

    def set_facecolor(self,color):
        self._circle.set_facecolor(color)

    def set_edgecolor(self,color):
        self._circle.set_edgecolor(color)

    def get_position(self):
        return self._x,self._y

    def get_radius(self):
        return self._radius

class InhLinearFiber():

    def __init__(self, startPos, endPos, width=2,synapseRadius=0.075):
        self._startPos = startPos
        self._endPos = endPos
        self._width = width
        self._synapseRadius = synapseRadius
        self.defaultColor = "#888888"

    def draw(self,ax):
        self._plot = ax.plot([self._startPos[0],self._endPos[0]],[self._startPos[1],self._endPos[1]],color=self.defaultColor,linewidth=self._width)
        self._circle = plt.Circle(self._endPos, radius=self._synapseRadius, edgecolor=self.defaultColor, facecolor=self.defaultColor)
        ax.add_patch(self._circle)

    def set_color(self,color):
        self._plot[0].set_color(color)
        self._circle.set_edgecolor(color)
        self._circle.set_facecolor(color)

class ExcLinearFiber():

    def __init__(self, startPos, endPos, endPosSyn1, endPosSyn2, width=2,synapseWidth=2):
        self._startPos = startPos
        self._endPos = endPos
        self._endPosSyn1 = endPosSyn1
        self._endPosSyn2 = endPosSyn2
        self._width = width
        self._synapseWidth = synapseWidth
        self.defaultColor = "#888888"

    def draw(self,ax):
        self._plot = ax.plot([self._startPos[0],self._endPos[0]],[self._startPos[1],self._endPos[1]],color=self.defaultColor,linewidth=self._width)
        self._syn1 = ax.plot([self._endPos[0],self._endPosSyn1[0]],[self._endPos[1],self._endPosSyn1[1]],color=self.defaultColor,linewidth=self._synapseWidth)
        self._syn2 = ax.plot([self._endPos[0],self._endPosSyn2[0]],[self._endPos[1],self._endPosSyn2[1]],color=self.defaultColor,linewidth=self._synapseWidth)

    def set_color(self,color):
        self._plot[0].set_color(color)
        self._syn1[0].set_color(color)
        self._syn2[0].set_color(color)

class ExcBezierFiber():

    def __init__(self, startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2, width=2,synapseWidth=2):
        self._startPos = startPos
        self._startPosCurv = startPosCurv
        self._endPos = endPos
        self._endPosCurv = endPosCurv
        self._endPosSyn1 = endPosSyn1
        self._endPosSyn2 = endPosSyn2
        self._width = width
        self._synapseWidth = synapseWidth
        self.defaultColor = "#888888"

    def draw(self,ax):
        verts = [self._endPos,   # P0
            self._endPosCurv,       # P1
            self._startPosCurv,     # P2
            self._startPos]      # P3
        codes = [Path.MOVETO,
                 Path.CURVE4,
                 Path.CURVE4,
                 Path.CURVE4]
        path = Path(verts, codes)
        self._patch = patches.PathPatch(path, facecolor='none', edgecolor=self.defaultColor, linewidth=self._width)
        ax.add_patch(self._patch)

        self._syn1 = ax.plot([self._endPos[0],self._endPosSyn1[0]],[self._endPos[1],self._endPosSyn1[1]],color=self.defaultColor,linewidth=self._synapseWidth)
        self._syn2 = ax.plot([self._endPos[0],self._endPosSyn2[0]],[self._endPos[1],self._endPosSyn2[1]],color=self.defaultColor,linewidth=self._synapseWidth)

    def set_color(self,color):
        self._patch.set_edgecolor(color)
        self._syn1[0].set_color(color)
        self._syn2[0].set_color(color)

class musclespindles_network_plotter():

    def __init__(self,extMusc="LEFT_LG",flexMusc="LEFT_TA",movieParam=None, labels=True, rosPublish=False):
        self._extMuscle = extMusc
        self._flexMuscle = flexMusc
        self._labels = labels

        self._ros_publisher = None
        if rosPublish:
            plt.switch_backend('Agg')
            self._ros_publisher = rospy.Publisher('/mouse_locomotion/muscle_spindles_network', Image, queue_size=1)
            self._cv_bridge = CvBridge()

        self._movieParam = movieParam
        self._colors = {self._extMuscle:"#f91f1f",self._flexMuscle:"#0085ff"}
        self._draw_network()

        if self._movieParam is not None:
            folder = self._movieParam['folder']
            fileName = self._movieParam['name']
            fps = self._movieParam['fps']
            FFMpegWriter = animation.writers['ffmpeg']
            self._writer = FFMpegWriter(fps=fps)
            self._writer.setup(self._fig, folder+fileName+".mp4", 200)

        if not rosPublish:
            plt.show(block=False)
            plt.pause(.1)


    def _draw_network(self):
        self._fig, ax = plt.subplots(1, figsize=(6,6))

        xMax = yMax = 10.
        xMin = yMin = 0.
        ax.set_xlim([xMin-1,xMax+1])
        ax.set_ylim([yMin-1,yMax+1])
        ax.set_axis_off()


        self._somas = {}
        self._somas[self._extMuscle] = {}
        self._somas[self._flexMuscle] = {}
        self._somas[self._extMuscle]['Mn'] = Neuron(xMax*0.65,yMax*0.4)
        self._somas[self._flexMuscle]['Mn'] = Neuron(xMax*0.35,yMax*0.4)
        self._somas[self._extMuscle]['IaInt'] = Neuron(xMax*0.7,yMax*0.75)
        self._somas[self._flexMuscle]['IaInt'] = Neuron(xMax*0.3,yMax*0.75)
        self._somas[self._extMuscle]['IIExInt'] = Neuron(xMax*0.8,yMax*0.575)
        self._somas[self._flexMuscle]['IIExInt'] = Neuron(xMax*0.2,yMax*0.575)
        self._somas[self._extMuscle]['Iaf'] = Neuron(xMax*0.975,yMax*0.2)
        self._somas[self._flexMuscle]['Iaf'] = Neuron(xMax*0.025,yMax*0.2)
        self._somas[self._extMuscle]['IIf'] = Neuron(xMax*0.9,yMax*0.2)
        self._somas[self._flexMuscle]['IIf'] = Neuron(xMax*0.1,yMax*0.2)

        for muscle in self._somas:
            for cellType in self._somas[muscle]:
                self._somas[muscle][cellType].draw(ax)
                self._somas[muscle][cellType].set_edgecolor(self._colors[muscle])

        if self._labels:
            # Create titles
            self._titles = []
            self._titles.append(ax.text(xMax*0.4,yMax*0.9, "Flexor network", verticalalignment='bottom', horizontalalignment='right',
            color="#0085ff", fontsize=12, weight='bold'))
            self._titles.append(ax.text(xMax*0.6, yMax*0.9, "Extensor network", verticalalignment='bottom', horizontalalignment='left',
            color="#f91f1f", fontsize=12, weight='bold'))

            # Create labels and firing rate text
            self._firingRates = {}
            self._firingRates[self._extMuscle] = {}
            self._firingRates[self._flexMuscle] = {}
            shiftX = xMax*0.05
            shiftY = yMax*0.025

            x,y = self._somas[self._flexMuscle]['Iaf'].get_position()
            self._firingRates[self._flexMuscle]['Iaf'] = ax.text(x-shiftX, y-shiftY, 'Iaf:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='right', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._flexMuscle]['IIf'].get_position()
            self._firingRates[self._flexMuscle]['IIf'] = ax.text(x+shiftX, y-shiftY, 'IIf:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='left', color="#191919", fontsize=8, weight='demibold', style='italic')

            x,y = self._somas[self._extMuscle]['Iaf'].get_position()
            self._firingRates[self._extMuscle]['Iaf'] = ax.text(x+shiftX, y-shiftY, 'Iaf:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='left', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._extMuscle]['IIf'].get_position()
            self._firingRates[self._extMuscle]['IIf'] = ax.text(x-shiftX, y-shiftY, 'IIf:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='right', color="#191919", fontsize=8, weight='demibold', style='italic')

            shiftY = yMax*0.05
            x,y = self._somas[self._flexMuscle]['Mn'].get_position()
            self._firingRates[self._flexMuscle]['Mn'] = ax.text(x, y+shiftY, 'Mn:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='center', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._flexMuscle]['IIExInt'].get_position()
            self._firingRates[self._flexMuscle]['IIExInt'] = ax.text(x, y+shiftY, 'IIExInt:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='left', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._flexMuscle]['IaInt'].get_position()
            self._firingRates[self._flexMuscle]['IaInt'] = ax.text(x, y+shiftY, 'IaInt:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='center', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._extMuscle]['Mn'].get_position()
            self._firingRates[self._extMuscle]['Mn'] = ax.text(x, y+shiftY, 'Mn:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='center', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._extMuscle]['IIExInt'].get_position()
            self._firingRates[self._extMuscle]['IIExInt'] = ax.text(x, y+shiftY, 'IIExInt:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='right', color="#191919", fontsize=8, weight='demibold', style='italic')
            x,y = self._somas[self._extMuscle]['IaInt'].get_position()
            self._firingRates[self._extMuscle]['IaInt'] = ax.text(x, y+shiftY, 'IaInt:\n0 Hz', verticalalignment='bottom',\
                horizontalalignment='center', color="#191919", fontsize=8, weight='demibold', style='italic')


        # Create axons
        self._fibers = {}
        self._fibers[self._extMuscle] = {}
        self._fibers[self._flexMuscle] = {}
        synShift = -0.075
        startPos,endPos1 = self._compute_fiber_position(self._somas[self._extMuscle]['Mn'],(xMax*0.65,yMax*0.1))
        startPos,endPos2 = self._compute_fiber_position(self._somas[self._extMuscle]['Mn'],(xMax*0.65,yMax*0.1+synShift))
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(startPos,endPos2)
        self._fibers[self._extMuscle]['Mn'] = [ExcLinearFiber(startPos, endPos1, endPosSyn1, endPosSyn2)]

        startPos,endPos1 = self._compute_fiber_position(self._somas[self._flexMuscle]['Mn'],(xMax*0.35,yMax*0.1))
        startPos,endPos2 = self._compute_fiber_position(self._somas[self._flexMuscle]['Mn'],(xMax*0.35,yMax*0.1+synShift))
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(startPos,endPos2)
        self._fibers[self._flexMuscle]['Mn'] = [ExcLinearFiber(startPos, endPos1, endPosSyn1, endPosSyn2)]

        startPos,endPos = self._compute_fiber_position(self._somas[self._extMuscle]['IaInt'],self._somas[self._flexMuscle]['Mn'])
        self._fibers[self._extMuscle]['IaInt'] = [InhLinearFiber(startPos,endPos)]
        startPos,endPos = self._compute_fiber_position(self._somas[self._extMuscle]['IaInt'],self._somas[self._flexMuscle]['IaInt'],0.1)
        self._fibers[self._extMuscle]['IaInt'].append(InhLinearFiber(startPos,endPos))
        startPos,endPos = self._compute_fiber_position(self._somas[self._flexMuscle]['IaInt'],self._somas[self._extMuscle]['Mn'])
        self._fibers[self._flexMuscle]['IaInt'] = [InhLinearFiber(startPos,endPos)]
        startPos,endPos = self._compute_fiber_position(self._somas[self._flexMuscle]['IaInt'],self._somas[self._extMuscle]['IaInt'],-0.1)
        self._fibers[self._flexMuscle]['IaInt'].append(InhLinearFiber(startPos,endPos))

        startPos,endPos1 = self._compute_fiber_position(self._somas[self._extMuscle]['IIExInt'],self._somas[self._extMuscle]['Mn'])
        startPos,endPos2 = self._compute_fiber_position(self._somas[self._extMuscle]['IIExInt'],self._somas[self._extMuscle]['Mn'],offsetY=0,offsetFiberDir=-0.05-synShift)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(startPos,endPos2)
        self._fibers[self._extMuscle]['IIExInt'] = [ExcLinearFiber(startPos, endPos1, endPosSyn1, endPosSyn2)]

        startPos,endPos1 = self._compute_fiber_position(self._somas[self._flexMuscle]['IIExInt'],self._somas[self._flexMuscle]['Mn'])
        startPos,endPos2 = self._compute_fiber_position(self._somas[self._flexMuscle]['IIExInt'],self._somas[self._flexMuscle]['Mn'],offsetY=0,offsetFiberDir=-0.05-synShift)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(startPos,endPos2)
        self._fibers[self._flexMuscle]['IIExInt'] = [ExcLinearFiber(startPos, endPos1, endPosSyn1, endPosSyn2)]



        # Create afferent fibers
        yShift = 0.1
        xPos = 0.025
        xShift = 0.05*xMax
        startPos = (self._somas[self._flexMuscle]['Iaf'].get_position()[0],self._somas[self._flexMuscle]['Iaf'].get_position()[1]-0.1*yMax)
        startPosCurv = (startPos[0],self._somas[self._flexMuscle]['IaInt'].get_position()[1]+yShift)
        endPos = (self._somas[self._flexMuscle]['IaInt'].get_position()[0]-xShift,self._somas[self._flexMuscle]['IaInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],self._somas[self._flexMuscle]['IaInt'].get_position()[1]+yShift)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]-synShift,endPos[1]))
        self._fibers[self._flexMuscle]['Iaf'] = [ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2)]

        startPosCurv = (startPos[0],self._somas[self._flexMuscle]['Mn'].get_position()[1])
        endPos = (self._somas[self._flexMuscle]['Mn'].get_position()[0]-xShift,self._somas[self._flexMuscle]['Mn'].get_position()[1])
        endPosCurv = (startPos[0],self._somas[self._flexMuscle]['Mn'].get_position()[1])
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]-synShift,endPos[1]))
        self._fibers[self._flexMuscle]['Iaf'].append(ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2))

        yShift = -0.25
        xPos = 0.1
        startPos = (self._somas[self._flexMuscle]['IIf'].get_position()[0],self._somas[self._flexMuscle]['IIf'].get_position()[1]-0.1*yMax)
        startPosCurv = (startPos[0],self._somas[self._flexMuscle]['IaInt'].get_position()[1]*4/5)
        endPos = (self._somas[self._flexMuscle]['IaInt'].get_position()[0]-xShift,self._somas[self._flexMuscle]['IaInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],(self._somas[self._flexMuscle]['IaInt'].get_position()[1]+yShift)*9/10)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]-synShift,endPos[1]+0.04))
        self._fibers[self._flexMuscle]['IIf'] = [ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2)]

        yShift = -0.5
        startPosCurv = (startPos[0],self._somas[self._flexMuscle]['IIExInt'].get_position()[1]*3/5)
        xShift = 0.025*xMax
        endPos = (self._somas[self._flexMuscle]['IIExInt'].get_position()[0]-xShift,self._somas[self._flexMuscle]['IIExInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],(self._somas[self._flexMuscle]['IIExInt'].get_position()[1]+yShift)*4/5)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]-synShift,endPos[1]+0.15))
        self._fibers[self._flexMuscle]['IIf'].append(ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2))

        xShift = -0.05*xMax
        yShift = 0.1
        xPos = 0.975
        startPos = (self._somas[self._extMuscle]['Iaf'].get_position()[0],self._somas[self._extMuscle]['Iaf'].get_position()[1]-0.1*yMax)
        startPosCurv = (startPos[0],self._somas[self._extMuscle]['IaInt'].get_position()[1]+yShift)
        endPos = (self._somas[self._extMuscle]['IaInt'].get_position()[0]-xShift,self._somas[self._extMuscle]['IaInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],self._somas[self._extMuscle]['IaInt'].get_position()[1]+yShift)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]+synShift,endPos[1]))
        self._fibers[self._extMuscle]['Iaf'] = [ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2)]

        startPosCurv = (startPos[0],self._somas[self._extMuscle]['Mn'].get_position()[1])
        endPos = (self._somas[self._extMuscle]['Mn'].get_position()[0]-xShift,self._somas[self._extMuscle]['Mn'].get_position()[1])
        endPosCurv = (startPos[0],self._somas[self._extMuscle]['Mn'].get_position()[1])
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]+synShift,endPos[1]))
        self._fibers[self._extMuscle]['Iaf'].append(ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2))

        yShift = -0.25
        xPos = 0.9
        startPos = (self._somas[self._extMuscle]['IIf'].get_position()[0],self._somas[self._extMuscle]['IIf'].get_position()[1]-0.1*yMax)
        startPosCurv = (startPos[0],self._somas[self._extMuscle]['IaInt'].get_position()[1]*4/5)
        endPos = (self._somas[self._extMuscle]['IaInt'].get_position()[0]-xShift,self._somas[self._extMuscle]['IaInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],(self._somas[self._extMuscle]['IaInt'].get_position()[1]+yShift)*9/10)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]+synShift,endPos[1]+0.04))
        self._fibers[self._extMuscle]['IIf'] = [ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2)]

        yShift = -0.5
        startPosCurv = (startPos[0],self._somas[self._extMuscle]['IIExInt'].get_position()[1]*3/5)
        xShift = -0.025*xMax
        endPos = (self._somas[self._extMuscle]['IIExInt'].get_position()[0]-xShift,self._somas[self._extMuscle]['IIExInt'].get_position()[1]+yShift)
        endPosCurv = (startPos[0],(self._somas[self._extMuscle]['IIExInt'].get_position()[1]+yShift)*4/5)
        endPosSyn1, endPosSyn2 = self._compute_excsyn_position(endPosCurv,(endPos[0]+synShift,endPos[1]+0.15))
        self._fibers[self._extMuscle]['IIf'].append(ExcBezierFiber(startPos, startPosCurv, endPos, endPosCurv, endPosSyn1, endPosSyn2))

        for muscle in self._fibers:
            for cellType in self._fibers[muscle]:
                for fiber in self._fibers[muscle][cellType]:
                    fiber.draw(ax)

    def _compute_excsyn_position(self,startPos,endPos,synDim=0.075):
        debug = False
        # debug = True
        if (endPos[0]-startPos[0])==0: m2=-9999999
        elif (endPos[1]-startPos[1])==0: m2=0.00000001
        else: m2 = (endPos[1]-startPos[1])/(endPos[0]-startPos[0])

        m2_90 = -1./m2
        b2_90 = endPos[1] -m2_90*endPos[0]

        dx = np.sqrt(synDim**2/(1+m2_90**2))
        x1 = endPos[0]+dx
        y1 = m2_90*x1 + b2_90
        x2 = endPos[0]-dx
        y2 = m2_90*x2+b2_90

        if debug:
            x = np.arange(0,10,0.1)
            y = m2_90*x+b2_90
            plt.gca().plot(x,y)
            plt.gca().plot([startPos[0],endPos[0]],[startPos[1],endPos[1]])
        return (x1,y1),(x2,y2)

    def _compute_fiber_position(self,startNeuron,endNeuron,offsetY=0,offsetFiberDir=-0.05):
        startPos = startNeuron.get_position()
        if isinstance(endNeuron,Neuron):
            endPosTemp = endNeuron.get_position()
            xDist = endPosTemp[0]-startPos[0]
            yDist = endPosTemp[1]-startPos[1]

            if (endPosTemp[0]-startPos[0])==0: m=-9999999
            elif (endPosTemp[1]-startPos[1])==0: m=0.00000001
            else: m = (endPosTemp[1]-startPos[1])/(endPosTemp[0]-startPos[0])
            dx = np.sign(offsetFiberDir)*abs(np.sqrt(offsetFiberDir**2/(1+m**2)))
            dy = np.sign(offsetFiberDir)*abs(dx*m)

            shift=0.05
            if xDist == 0 and yDist != 0:
                endPos = startPos[0],np.sign(yDist)*dy+startPos[1]+yDist*(abs(yDist)-endNeuron.get_radius()*np.sqrt(2)-shift)/abs(yDist)
            elif yDist == 0 and xDist != 0:
                endPos = np.sign(xDist)*dx+startPos[0]+xDist*(abs(xDist)-endNeuron.get_radius()*np.sqrt(2)-shift)/abs(xDist),startPos[1]
            elif xDist != 0 and yDist != 0:
                endPos = np.sign(xDist)*dx+startPos[0]+xDist*(abs(xDist)-endNeuron.get_radius()-shift)/abs(xDist),\
                    np.sign(yDist)*dy+startPos[1]+ yDist*(abs(yDist)-endNeuron.get_radius()-shift)/abs(yDist)
            else: raise(Exception('invalid end position'))
        else: endPos = endNeuron
        endPos = endPos[0],endPos[1]+offsetY
        startPos = startPos[0],startPos[1]+offsetY
        return startPos,endPos

    def update_activity(self,activityColors,firingRates=None):
        for muscle in activityColors:
            for cellType in activityColors[muscle]:
                self._somas[muscle][cellType].set_facecolor(activityColors[muscle][cellType])
                for fiber in self._fibers[muscle][cellType]:
                    fiber.set_color(activityColors[muscle][cellType])

        if firingRates is not None:
            for muscle in firingRates:
                for cellType in firingRates[muscle]:
                    self._firingRates[muscle][cellType].set_text("%s:\n%d Hz"%(cellType,firingRates[muscle][cellType]))

        if self._movieParam is not None:
            self._writer.grab_frame()

        if self._ros_publisher:
          self._fig.canvas.draw()
          img_data = np.fromstring(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
          img_data = img_data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
          ros_img = self._cv_bridge.cv2_to_imgmsg(img_data, 'rgb8')
          self._ros_publisher.publish(ros_img)
        else:
          plt.pause(0.01)



    def end_plotting(self):
        if self._movieParam is not None:
            self._writer.finish()


if __name__ == '__main__':
    mnp = musclespindles_network_plotter()
    plt.show()

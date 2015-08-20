# -*- coding: utf-8 -*-
## @package som_cm.som
#
#  Implementation of SOM.
#  @author      tody
#  @date        2015/08/14

from docopt import docopt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
from som_cm.io_util.image import loadRGB
from som_cm.np.norm import normVectors
from som_cm.datasets.google_image import loadData, dataFile
from som_cm.cv.image import to32F
from som_cm.core.color_samples import Hist3D
from som_cm.plot.window import showMaximize

_root_dir = os.path.dirname(__file__)


## Result directory for SOM results.
def resultDir():
    result_dir = os.path.abspath(os.path.join(_root_dir, "../results"))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

## SOM parameter.
class SOMParam:
    #  @param h           image grid size.
    #  @param L0          initial parameter for learning restraint.
    #  @param lmbd        iteration limit.
    #  @param dimensoin   target dimension for SOM.
    def __init__(self, h=32, L0=0.16, lmbd=0.6, sigma0=0.3, dimension=2):
        self.h = h
        self.L0 = L0
        self.lmbd = lmbd
        self.sigma0 = sigma0
        self.dimension = dimension

## Implementation of SOM.
#
#  Batch SOM with numpy functions.
#  - Compute nodes as n x 3 vector.
#  - Avoid the loops for x and y.
#  - xy coordinates are cached as n x 2 vector.
class SOM:
    ## Constructor
    #  @param samples  training samples.
    #  @param param    SOM parameter.
    def __init__(self, samples, param=SOMParam()):
        self._h = param.h
        self._dimension = param.dimension
        self._samples = samples
        self._L0 = param.L0

        self._nodes = self._initialNode(param.h, param.dimension)

        num_samples = self.numSamples()
        self._lmbd = param.lmbd * num_samples

        self._sigma0 = param.sigma0 * param.h

        self._computePositions(param.h, param.dimension)

        self._t = 0

    ## Return the number of training samples.
    def numSamples(self):
        return len(self._samples)

    ## Return the current node image.
    def nodeImage(self):
        if self._dimension == 1:
            return self._nodeImage1D()
        else:
            return self._nodeImage2D()

    def _nodeImage1D(self):
        h = 10
        w = self._h
        node_image = np.zeros((h, w, 3))
        for y in range(h):
            node_image[y, :, :] = self._nodes[:, :]
        return node_image

    def _nodeImage2D(self):
        return self._nodes.reshape(self._h, self._h, 3)

    ## Return the current time step t.
    def currentStep(self):
        return self._t

    ## Return if the training is finished.
    def finished(self):
        return self._t == self.numSamples()

    ## Process all training process.
    def trainAll(self):
        while self._t < len(self._samples):
            self._train(self._t)
            self._t += 1

    ## Process training step t to t+1.
    def trainStep(self):
        if self._t < len(self._samples):
            self._train(self._t)
            self._t += 1

    ## Initial node.
    def _initialNode(self, h, dimension):
        if dimension == 1:
            return self._initialNode1D(h)
        else:
            return self._initialNode2D(h)

    def _initialNode1D(self, h):
        return np.random.rand(h, 3)

    def _initialNode2D(self, h):
        return np.random.rand(h, h, 3).reshape(-1, 3)

    ## Compute position.
    def _computePositions(self, h, dimension):
        if dimension == 1:
            self._computePositions1D(h)
        else:
            self._computePositions2D(h)

    def _computePositions1D(self, h):
        x = np.arange(h)
        self._positions = x

    def _computePositions2D(self, h):
        x = np.arange(h)
        y = np.arange(h)
        xs, ys = np.meshgrid(x, y)
        xs = xs.flatten()
        ys = ys.flatten()
        self._positions = np.array([xs, ys]).T

    ## Train process.
    def _train(self, t):
        sample = self._samples[t]

        # bmu
        bmu_id = self._bmu(sample)
        bmu_position = self._positions[bmu_id]

        # update weight
        D = normVectors(self._positions - bmu_position)
        L = self._learningRestraint(t)
        T = self._neighborhoodFunction(t, D)

        # update nodes
        for ci in range(3):
            self._nodes[:, ci] += L * T * (sample[ci] - self._nodes[:, ci])

    ## BMU: best matching unit.
    #  Return the unit of minimum distance from the sample.
    def _bmu(self, sample):
        norms = normVectors(self._nodes - sample)
        bmu_id = np.argmin(norms)
        return bmu_id

    ## Neighborhood function: exp (-D^2 / 2 sigma^2)
    def _neighborhoodFunction(self, t, D):
        sigma = self._sigma0 * np.exp(-t / self._lmbd)
        Theta = np.exp(-D ** 2 / (2 * sigma ** 2))
        return Theta

    ## Learning restraint: L0 exp (-t / lambda)
    def _learningRestraint(self, t):
        return self._L0 * np.exp(-t / self._lmbd)


## Plotting class with matplot.
class SOMPlot:
    ## Constructor
    #  @param samples training samples.
    #  @param param    SOM parameter.
    def __init__(self, som):
        self._som = som
        self._node_image = None
        self._plot3d = None
        self._step_text = None

    ## Return the updated image.
    def updateImage(self):
        node_image = self._som.nodeImage()
        if self._node_image is None:
            self._node_image = plt.imshow(node_image)

        else:
            self._node_image.set_array(node_image)

        return self._node_image

    ## Return the current step status.
    def updateStepText(self):
        if self._step_text is None:
            self._step_text = plt.text(1, 1, '', fontsize=15)

        else:
            if self._som.finished():
                self._step_text.set_text('')
            else:
                self._step_text.set_text('step: %s' % self._som.currentStep())

        return self._step_text

    def plot3D(self, ax):
        node_image = self._som.nodeImage()
        colors = node_image.reshape(-1, 3)
        plot3d = ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2],
                    color=colors)

        ax.set_xlabel('R', x=10, y=10)
        ax.set_ylabel('G')
        ax.set_zlabel('B')

        ax.set_zlim3d([0.0, 1.0])
        ax.set_ylim3d([0.0, 1.0])
        ax.set_xlim3d([0.0, 1.0])

        ax.set_xticks(np.linspace(0.2, 0.8, 2))
        ax.set_yticks(np.linspace(0.2, 0.8, 2))
        ax.set_zticks(np.linspace(0.2, 0.8, 2))
        return plot3d

    ## Animation function for FuncAnimation.
    def trainAnimation(self, *args):
        image = self.updateImage()
        text = self.updateStepText()

        self._som.trainStep()

        return [image, text]


def setupSOM(image_file, random_seed=100, num_samples=1000):
    np.random.seed(random_seed)
    C_8U = loadRGB(image_file)
    C_32F = to32F(C_8U)

    hist3D = Hist3D(C_32F)
    color_samples = hist3D.colorSamples()

    random_ids = np.random.randint(len(color_samples) - 1, size=num_samples)
    samples = color_samples[random_ids]

    param1D = SOMParam(h=64, dimension=1)
    som1D = SOM(samples, param1D)

    param2D = SOMParam(h=32, dimension=2)
    som2D = SOM(samples, param2D)
    return C_32F, som1D, som2D


def runSOMAnimation(image_name, C_32F, som1D, som2D):
    fig = plt.figure()
    plt.title("SOM-Color Manifolds")
    plt.subplot(131)
    plt.title("%s" % (image_name))
    plt.imshow(C_32F)
    plt.axis('off')

    plt.subplot(132)
    plt.title("SOM 1D")
    som1D_plot = SOMPlot(som1D)
    ani1D = animation.FuncAnimation(fig, som1D_plot.trainAnimation, interval=0, blit=True)
    plt.axis('off')

    plt.subplot(133)
    plt.title("SOM 2D")
    som2D_plot = SOMPlot(som2D)
    ani2D = animation.FuncAnimation(fig, som2D_plot.trainAnimation, interval=0, blit=True)
    plt.axis('off')

    showMaximize()


def runSOMResult(image_name, C_32F, som1D, som2D):
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
    plt.title("SOM-Color Manifolds")
    plt.subplot(231)
    plt.title("%s" % (image_name))
    plt.imshow(C_32F)
    plt.axis('off')

    print "  - Train 1D"
    som1D.trainAll()

    print "  - Train 2D"
    som2D.trainAll()

    som1D_plot = SOMPlot(som1D)
    som2D_plot = SOMPlot(som2D)
    plt.subplot(232)
    plt.title("SOM 1D")
    som1D_plot.updateImage()
    plt.axis('off')

    plt.subplot(233)
    plt.title("SOM 2D")
    som2D_plot.updateImage()
    plt.axis('off')

    ax1D = fig.add_subplot(235, projection='3d')
    plt.title("1D in 3D")
    som1D_plot.plot3D(ax1D)

    ax2D = fig.add_subplot(236, projection='3d')
    plt.title("2D in 3D")
    som2D_plot.plot3D(ax2D)

    plt.savefig(os.path.join(resultDir(), image_name + ".png"))
    #showMaximize()


def runSOMResults(data_names, data_ids):
    for data_name in data_names:
        print "SOM: %s" % data_name
        for data_id in data_ids:
            print "Data ID: %s" % data_id
            image_file = dataFile(data_name, data_id)
            image_name = os.path.basename(image_file)
            image_name = os.path.splitext(image_name)[0]
            C_32F, som1D, som2D = setupSOM(image_file)
            runSOMResult(image_name, C_32F, som1D, som2D)

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    data_ids = [0, 1, 2]

    runSOMResults(data_names, data_ids)

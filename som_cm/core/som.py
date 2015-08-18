
# -*- coding: utf-8 -*-
## @package som_cm.som
#
#  Implementation of SOM.
#  @author      tody
#  @date        2015/08/14

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

from som_cm.np.norm import normVectors
from som_cm.datasets.google_image import loadData
from som_cm.cv.image import to32F


## SOM parameter.
class SOMParam:
    #  @param h       image grid size.
    #  @param L0      initial parameter for learning restraint.
    #  @param lmbd    iteration limit.
    def __init__(self, h=32, L0=0.06, lmbd=0.5, sigma0=0.3):
        self.h = h
        self.L0 = L0
        self.lmbd = lmbd
        self.sigma0 = sigma0


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
        self._nodes = self._initialNode(param.h)
        self._h = param.h
        self._samples = samples
        self._L0 = param.L0

        num_samples = self.numSamples()
        self._lmbd = param.lmbd * num_samples

        self._sigma0 = param.sigma0 * param.h

        x = np.arange(param.h)
        y = np.arange(param.h)
        xs, ys = np.meshgrid(x, y)
        xs = xs.flatten()
        ys = ys.flatten()
        self._xy = np.array([xs, ys]).T

        self._t = 0

    ## Return the number of training samples.
    def numSamples(self):
        return len(self._samples)

    ## Return the current node image.
    def nodeImage(self):
        return self._nodes.reshape(self._h, self._h, 3)

    ## Return the current time step t.
    def currentStep(self):
        return self._t

    ## Return if the training is finished.
    def finished(self):
        return self._t == self.numSamples()

    ## Process training step t to t+1.
    def trainStep(self):
        if self._t < len(self._samples):
            self._train(self._t)
            self._t += 1

    ## Initial node.
    def _initialNode(self, h):
        return np.random.rand(h, h, 3).reshape(-1, 3)

    ## Train process.
    def _train(self, t):
        sample = self._samples[t]

        # bmu
        bmu_id = self._bmu(sample)
        bmu_xy = self._xy[bmu_id]

        # update weight
        D = normVectors(self._xy - bmu_xy)
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
    def __init__(self, samples, param=SOMParam()):
        self._som = SOM(samples, param)
        self._node_image = None
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
                self._step_text.set_text('finished')
            else:
                self._step_text.set_text('step: %s' % self._som.currentStep())

        return self._step_text

    ## Animation function for FuncAnimation.
    def trainAnimation(self, *args):
        image = self.updateImage()
        text = self.updateStepText()

        self._som.trainStep()

        return [image, text]

if __name__ == '__main__':
    np.random.seed(100)
    num_samples = 2000
    #samples = np.random.rand(num_samples, 3)

    C_8U = loadData(data_name="banana", i=3)
    C_32F = to32F(C_8U)
    print C_8U.shape

    C = C_32F.reshape(-1, 3)
    print np.max(C)

    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(C_8U)

    br_dist = normVectors(C-np.array([1.0, 1.0, 1.0]))
    C = C[br_dist > 0.05]

    dr_dist = normVectors(C-np.array([0, 0, 0]))
    #C = C[dr_dist > 0.1]

    random_ids = np.random.randint(len(C) - 1, size=num_samples)

    samples = C[random_ids]
    samples_sparse = samples[::len(samples) / 64]
    print samples_sparse.shape
    samples_image = np.zeros((64, 20, 3))

    for ri in range(20):
        samples_image[:,ri,:] = samples_sparse[:64,:]

    plt.subplot(132)
    plt.imshow(samples_image)

    param = SOMParam(h=32)
    som = SOMPlot(samples, param)
    plt.subplot(133)

    ani = animation.FuncAnimation(fig, som.trainAnimation, interval=5, blit=True)
    plt.show()

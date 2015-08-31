# -*- coding: utf-8 -*-
## @package som_cm.results.animation
#
#  som_cm.results.animation utility package.
#  @author      tody
#  @date        2015/08/31

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from som_cm.core.hist_3d import Hist3D
from som_cm.core.som import SOMParam, SOM, SOMPlot
from som_cm.io_util.image import loadRGB
from som_cm.plot.window import showMaximize
from som_cm.results.results import batchResults


## Compute SOM in 1D and 2D for the target image.
def setupSOM(image, random_seed=100, num_samples=1000):
    np.random.seed(random_seed)

    hist3D = Hist3D(image)
    color_samples = hist3D.colorCoordinates()

    random_ids = np.random.randint(len(color_samples) - 1, size=num_samples)
    samples = color_samples[random_ids]

    param1D = SOMParam(h=64, dimension=1)
    som1D = SOM(samples, param1D)

    param2D = SOMParam(h=32, dimension=2)
    som2D = SOM(samples, param2D)
    return som1D, som2D


## Compute SOM selection result for the image file.
#
#  Note:
#  Current implementation causes a problem of Tkinter when destroy the figure canvas.
def animationResult(image_file):
    try:
        animationResultImp(image_file)
    except:
        print "Catch Tkinter Exception"


## Compute SOM selection result for the image file.
def animationResultImp(image_file):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]

    image = loadRGB(image_file)

    som1D, som2D = setupSOM(image)

    fig = plt.figure()
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)

    font_size = 15
    fig.suptitle("SOM-Color Manifolds (Animation)", fontsize=font_size)

    plt.subplot(131)
    plt.title("%s" % (image_name))
    plt.imshow(image)
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


## Compute SOM results for the given data names, ids.
def animationResults(data_names, data_ids):
    batchResults(data_names, data_ids, animationResult, "SOM (single image)")

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    data_ids = [0, 1, 2]

    animationResults(data_names, data_ids)
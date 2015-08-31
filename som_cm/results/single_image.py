# -*- coding: utf-8 -*-
## @package som_cm.results.som_single_image
#
#  som_cm.results.som_single_image utility package.
#  @author      tody
#  @date        2015/08/31

import os
import numpy as np
import matplotlib.pyplot as plt

from som_cm.io_util.image import loadRGB
from som_cm.results.results import batchResults, resultFile
from som_cm.core.hist_3d import Hist3D
from som_cm.core.som import SOMParam, SOM, SOMPlot
from som_cm.plot.window import showMaximize


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


## Compute palette selection result for the image file.
def singleImageResult(image_file):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]

    image = loadRGB(image_file)

    som1D, som2D = setupSOM(image)

    fig = plt.figure(figsize=(10, 7))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)

    font_size = 15
    fig.suptitle("SOM-Color Manifolds for Single Image", fontsize=font_size)

    plt.subplot(231)
    h, w = image.shape[:2]
    plt.title("Original Image: %s x %s" % (w, h), fontsize=font_size)
    plt.imshow(image)
    plt.axis('off')

    print "  - Train 1D"
    som1D.trainAll()

    print "  - Train 2D"
    som2D.trainAll()

    som1D_plot = SOMPlot(som1D)
    som2D_plot = SOMPlot(som2D)
    plt.subplot(232)
    plt.title("SOM 1D", fontsize=font_size)
    som1D_plot.updateImage()
    plt.axis('off')

    plt.subplot(233)
    plt.title("SOM 2D", fontsize=font_size)
    som2D_plot.updateImage()
    plt.axis('off')

    ax1D = fig.add_subplot(235, projection='3d')
    plt.title("1D in 3D", fontsize=font_size)
    som1D_plot.plot3D(ax1D)

    ax2D = fig.add_subplot(236, projection='3d')
    plt.title("2D in 3D", fontsize=font_size)
    som2D_plot.plot3D(ax2D)

    result_file = resultFile("%s_single" % image_name)
    plt.savefig(result_file)
    #showMaximize()


## Compute SOM results for the given data names, ids.
def singleImageResults(data_names, data_ids):
    batchResults(data_names, data_ids, singleImageResult, "SOM (single image)")

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    data_ids = [0, 1, 2]

    singleImageResults(data_names, data_ids)
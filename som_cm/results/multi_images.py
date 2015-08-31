# -*- coding: utf-8 -*-
## @package som_cm.results.multi_images
#
#  Demo for multi-images.
#  @author      tody
#  @date        2015/08/31

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from som_cm.datasets.google_image import dataFile
from som_cm.io_util.image import loadRGB
from som_cm.core.hist_3d import Hist3D
from som_cm.core.som import SOMParam, SOM, SOMPlot
from som_cm.results.results import resultFile, batchDataGroup


### Setup SOM in 1D and 2D for the target color samples.
def setupSOM(color_samples, random_seed=100, num_samples=1000):
    np.random.seed(random_seed)

    random_ids = np.random.randint(len(color_samples) - 1, size=num_samples)
    samples = color_samples[random_ids]

    param1D = SOMParam(h=64, dimension=1)
    som1D = SOM(samples, param1D)

    param2D = SOMParam(h=32, dimension=2)
    som2D = SOM(samples, param2D)
    return som1D, som2D


## Demo for the given data group.
def multiImagesResult(data_name, data_ids):
    num_cols = len(data_ids)
    num_rows = 2

    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)

    font_size = 15
    fig.suptitle("SOM-Color Manifolds for Multi Images", fontsize=font_size)

    color_samples = []
    col_id = 0
    for data_id in data_ids:
        image_file = dataFile(data_name, data_id)
        image_name = "%s_%s" % (data_name, data_id + 1)

        image = loadRGB(image_file)

        hist3D = Hist3D(image, num_bins=16)

        color_samples.extend(hist3D.colorCoordinates())

        plt.subplot2grid((num_rows, num_cols), (0, col_id))
        plt.title("%s" % image_name, fontsize=font_size)
        plt.imshow(image)
        plt.axis('off')

        col_id += 1

    color_samples = np.array(color_samples)
    print color_samples.shape

    som1D, som2D = setupSOM(color_samples)

    print "  - Train 1D"
    som1D.trainAll()

    print "  - Train 2D"
    som2D.trainAll()

    som1D_plot = SOMPlot(som1D)
    som2D_plot = SOMPlot(som2D)

    col_id = 1
    plt.subplot2grid((num_rows, num_cols), (1, col_id))
    plt.title("SOM 1D", fontsize=font_size)
    som1D_plot.updateImage()
    plt.axis('off')

    col_id += 1
    ax1D = plt.subplot2grid((num_rows, num_cols), (1, col_id),
                            projection='3d', aspect='equal')
    plt.title("1D in 3D", fontsize=font_size)
    som1D_plot.plot3D(ax1D)

    col_id += 1
    plt.subplot2grid((num_rows, num_cols), (1, col_id))
    plt.title("SOM 2D", fontsize=font_size)
    som2D_plot.updateImage()
    plt.axis('off')

    col_id += 1
    ax2D = plt.subplot2grid((num_rows, num_cols), (1, col_id),
                            projection='3d', aspect='equal')
    plt.title("2D in 3D", fontsize=font_size)
    som2D_plot.plot3D(ax2D)

    result_file = resultFile("%s_multi" % data_name)
    plt.savefig(result_file)


## Demo for the given data names, ids.
def multiImagesResults(data_names, data_ids):
    batchDataGroup(data_names, data_ids,
                   multiImagesResult, "SOM (multi images)")

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    data_ids = [0, 1, 2, 3, 4]

    multiImagesResults(data_names, data_ids)

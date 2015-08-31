# -*- coding: utf-8 -*-
## @package som_cm.main
#
#  Main functions.
#  @author      tody
#  @date        2015/08/19
from som_cm.datasets.google_image import createDatasets
from som_cm.results.single_image import singleImageResults
from som_cm.results.multi_images import multiImagesResults

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    num_images = 9

    createDatasets(data_names, num_images, update=False)
    data_ids = range(3)
    singleImageResults(data_names, data_ids)

    data_ids = range(5)
    multiImagesResults(data_names, data_ids)

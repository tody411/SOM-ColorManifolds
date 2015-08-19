# -*- coding: utf-8 -*-
## @package som_cm.main
#
#  Main functions.
#  @author      tody
#  @date        2015/08/19
from som_cm.datasets.google_image import createDatasets
from som_cm.core.som import runSOMResults

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    num_images = 3
    data_ids = range(3)

    createDatasets(data_names, num_images, update=False)
    runSOMResults(data_names, data_ids)

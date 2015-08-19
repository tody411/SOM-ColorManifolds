# -*- coding: utf-8 -*-
## @package som_cm.datasets.google_image
#
#  Google image datasets.
#
#  @author      tody
#  @date        2015/08/15


import json
import os
import urllib2

import cv2
import matplotlib.pyplot as plt

from som_cm.io.image import loadRGBA, loadRGB, saveRGB

_root_dir = os.path.dirname(__file__)


def dataDir(data_name):
    data_dir = os.path.join(_root_dir, data_name)
    return data_dir


def dataFiles(data_name):
    data_dir = dataDir(data_name)
    data_files = []
    for data_name in os.listdir(data_dir):
        if ".png" in data_name or ".jpg" in data_name:
            data_files.append(os.path.join(data_dir, data_name))
    return data_files


def loadData(data_name, i):
    data_files = dataFiles(data_name)

    if i >= len(data_files):
        return None

    data_file = data_files[i]

    return loadRGB(data_file)


def searchImages(keyword, num_images):

    url_list = []
    url = "http://ajax.googleapis.com/ajax/services/search/images?q={0}&v=1.0&rsz=large&start={1}"

    for i in range((num_images / 8) + 1):
        res = urllib2.urlopen(url.format(keyword, i * 8))
        data = json.load(res)
        url_list += [result["url"] for result in data["responseData"]["results"]]

    return url_list


def downloadImages(data_name, num_images):
    url_list = searchImages(data_name, num_images)
    data_dir = os.path.join(_root_dir, data_name)
    if os.path.exists(data_dir) == False:
        os.makedirs(data_dir)

    opener = urllib2.build_opener()

    for i in range(len(set(url_list))):
        try:
            fn, ext = os.path.splitext(url_list[i])
            req = urllib2.Request(url_list[i], headers={"User-Agent": "Magic Browser"})
            data_file = open(os.path.join(data_dir, "%s_%s%s" % (data_name, i, ext)), "wb")
            data_file.write(opener.open(req).read())
            data_file.close()
            print("Downloaded:"+str(i+1))
        except:
            continue

def resizeImages(data_name):
    data_files = dataFiles(data_name)

    for data_file in dataFiles(data_name):
        C_8U = loadRGB(data_file)
        if C_8U is None:
            os.remove(data_file)
            continue
        h, w = C_8U.shape[0:2]
        print h, w
        opt_scale = 800.0 / float(h)
        opt_scale = max(opt_scale, 800.0 / float(w))
        print opt_scale

        h_opt = int(opt_scale * h)
        w_opt = int(opt_scale * w)

        C_8U_small = cv2.resize(C_8U, (w_opt, h_opt))
        saveRGB(data_file, C_8U_small)

def testLoad(data_name="banana", i=0):
    C_8U = loadData(data_name, i)
    plt.imshow(C_8U)
    plt.show()


def testDownload(keyword="banana", num_images=16):
    downloadImages(keyword, num_images)

if __name__ == '__main__':
    #testDownload("sky")
    resizeImages("sky")
    #testLoad(data_name="banana", i=1)

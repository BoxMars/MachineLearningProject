import os
import sys
import time

import numpy
from skimage import io, feature, transform

class Data:
    BASE_DIR = ""
    DATA_DIR = ""
    ORIGINAL_DIR = ""
    TRAIN_DIR = ""
    TEST_DIR = ""
    CROP_DIR=""
    TEST_CROP_DIR=""
    def __init__(self):
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            ))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.ORIGINAL_DIR = os.path.join(self.DATA_DIR, 'original')
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.TEST_DIR = os.path.join(self.DATA_DIR, 'test')
        self.CROP_DIR=os.path.join(self.DATA_DIR,'crop')
        self.TEST_CROP_DIR = os.path.join(self.DATA_DIR, 'test_crop')

    def get_data(self, set_path):
        dir_list = os.listdir(set_path)
        print("Load data from "+ set_path)
        size=32
        res = []
        output = []
        # for i in range(1):
        for i in range(len(dir_list)):
            # print(bcolors.OKBLUE+dir_list[i]+bcolors.ENDC)
            label = int(dir_list[i])
            path = os.path.join(set_path, dir_list[i])
            for imagePath in os.listdir(path):
                print("\r", end="")
                print('...............' + path, end="")
                sys.stdout.flush()
                img = io.imread(os.path.join(path, imagePath), as_gray=True)
                img = transform.resize(img, (size, size))
                # edges = feature.canny(img, sigma=0.6)
                # res.append(edges)
                res.append(img)
                output.append(label)
            # print("\n")
        print("Data loading finish")

        res = numpy.array(res)
        num, nx, ny = res.shape
        res = res.reshape((num, nx * ny))
        output = numpy.array(output)
        numpy.random.seed(100)
        numpy.random.shuffle(res)
        numpy.random.seed(100)
        numpy.random.shuffle(output)
        return res, output

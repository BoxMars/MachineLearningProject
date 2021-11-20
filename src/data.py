import os
import numpy
from skimage import io, feature, transform


class Data:
    BASE_DIR = ""
    DATA_DIR = ""
    ORIGINAL_DIR = ""
    TRAIN_DIR = ""
    TEST_DIR = ""

    def __init__(self):
        self.BASE_DIR = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            ))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.ORIGINAL_DIR = os.path.join(self.DATA_DIR, 'original')
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, 'train')
        self.TEST_DIR = os.path.join(self.DATA_DIR, 'test')

    def get_data(self, set_path):
        dir_list = os.listdir(set_path)
        res = []
        output = []
        for i in range(len(dir_list)):
            label = int(dir_list[i])
            print(label)
            path = os.path.join(self.TRAIN_DIR, dir_list[i])
            for imagePath in os.listdir(path):
                img = io.imread(os.path.join(path, imagePath), as_gray=True)
                img = transform.resize(img, (32, 32))
                edges = feature.canny(img, sigma=0.6)
                res.append(edges)
                output.append(label)
        res = numpy.array(res)
        num, nx, ny = res.shape
        res = res.reshape((num, nx * ny))
        print(res.shape)
        output = numpy.array(output)
        return res, output

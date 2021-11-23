import time

from sklearn import neural_network
from sklearn.model_selection import ShuffleSplit, cross_val_score

from src.data import Data


def train_test(x, y):
    nn = neural_network.MLPClassifier(verbose=1)
    shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=5)
    scores = cross_val_score(nn, x, y, cv=shuffle)
    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))
    print("Finish training")


if __name__ == "__main__":
    start = time.time()
    data = Data()
    x,y=data.get_data(data.CROP_DIR)
    train_test(x,y)
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

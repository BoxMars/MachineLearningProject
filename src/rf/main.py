import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from src.data import Data


def train_test(x, y):
    print("Start training")
    rfc = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=90, verbose=1)
    shuffle = ShuffleSplit(train_size=.7, test_size=.2, n_splits=8)
    scores = cross_val_score(rfc, x, y, cv=shuffle)
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

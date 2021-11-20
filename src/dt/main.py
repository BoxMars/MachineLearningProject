import time

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from src.data import Data


def main():
    data = Data()
    x_train, y_train = data.get_data(data.TRAIN_DIR)
    x_test, y_test = data.get_data(data.TEST_DIR)
    rfc = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=90)
    rfc.fit(x_train,y_train)
    print('Running time: %s Seconds' % (end - start))
    print(rfc.score(x_test,y_test))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

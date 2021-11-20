from sklearn.svm import SVC
from src.data import Data
import time

def main():
    data = Data()
    x_train, y_train = data.get_data(data.TRAIN_DIR)
    x_test, y_test = data.get_data(data.TEST_DIR)
    clf = SVC()
    clf.fit(x_train,y_train)
    print(clf.score(x_test,y_test))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
from sklearn import decomposition, svm
from sklearn.model_selection import cross_val_score
from src.data import Data
import time


def main():
    data = Data()
    x_train, y_train = data.get_data(data.CROP_DIR)
    # x_test, y_test = data.get_data(data.TEST_DIR)
    pca = decomposition.IncrementalPCA()
    # pca.fit(x_train,y_train)
    # trainW = pca.transform(x_train)  # fit the training set
    trainW = pca.fit_transform(x_train)  # fit the training set
    pca2=decomposition.IncrementalPCA()
    trainW=pca2.fit_transform(trainW)
    svmclf = svm.SVC(kernel='rbf')
    scores = cross_val_score(svmclf, trainW, y_train, cv=3)
    print("cross valid kernel pca + svm =", scores)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

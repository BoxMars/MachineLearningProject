import time

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, ShuffleSplit
from src.utils.colors import bcolors
from src.data import Data


from sklearn.metrics import classification_report, accuracy_score, make_scorer

def classification_report_with_accuracy_score(y_true, y_pred):

    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

def main():
    data = Data()
    # x_train, y_train = data.get_data(data.TRAIN_DIR)
    # x_test, y_test = data.get_data(data.TEST_DIR)

    x,y=data.get_data(data.CROP_DIR)
    print(bcolors.OKCYAN + "Start training" + bcolors.ENDC)
    rfc = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=90)
    # rfc.fit(x_train, y_train)
    #strKFold = StratifiedKFold(n_splits=15)
    shufspl = ShuffleSplit(train_size=.7, test_size=.3, n_splits=5)
    scores = cross_val_score(rfc, x, y, cv=shufspl)

    print("Cross validation scores:{}".format(scores))
    print("Mean cross validation score:{:2f}".format(scores.mean()))

    nested_score = cross_val_score(rfc, X=x, y=y, scoring=make_scorer(classification_report_with_accuracy_score))
    print(nested_score)

    print(bcolors.OKCYAN + "Finish training" + bcolors.ENDC)



    # print(rfc.score(x_test, y_test))

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

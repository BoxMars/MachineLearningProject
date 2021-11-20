from sklearn import tree
from src.data import Data


def main():
    data = Data()

    x_train, y_train = data.get_data(data.TRAIN_DIR)
    x_test, y_test = data.get_data(data.TEST_DIR)

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    print(classifier.score(x_test, y_test))


if __name__ == "__main__":
    main()

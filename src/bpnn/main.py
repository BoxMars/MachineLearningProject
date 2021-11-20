from sklearn import neural_network
from src.data import Data

data=Data()

x_train, y_train = data.get_data(data.TRAIN_DIR)
x_test, y_test = data.get_data(data.TEST_DIR)

bpnn=neural_network.MLPClassifier()
bpnn.fit(x_train,y_train)
print(bpnn.score(x_test,y_test))


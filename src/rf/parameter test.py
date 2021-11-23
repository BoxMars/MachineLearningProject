import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.utils.colors import bcolors
from src.data import Data

data = Data()
min_estimators = 1
max_estimators = 250
train_scores = []
test_scores = []
x_train, y_train = data.get_data(data.CROP_DIR)
randomForest = RandomForestClassifier(n_estimators=160, n_jobs=-1, random_state=0)
for i in range(min_estimators, max_estimators + 1, 5):
        randomForest.set_params(random_state=i)
        rf=randomForest.fit(x_train, y_train)
        train_scores.append(np.mean(cross_val_score(rf, x_train, y_train, cv=3)))
        #test_scores.append(randomForest.score(X_test, y_test))

fig, ax = plt.subplots(dpi = 100)
ax.set_xlabel("estimators")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs estimators for training and testing sets")
ax.plot(range(min_estimators, max_estimators + 1, 5), train_scores, label="train",
        drawstyle="steps-post")
#ax.plot(range(min_estimators, max_estimators + 1, 5), test_scores, label="test",
#        drawstyle="steps-post")
ax.legend()
plt.show()








import time

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.utils.colors import bcolors
from src.data import Data

data = Data()
min_estimators = 1
max_estimators = 250
train_scores = []
test_scores = []
x_train, y_train = data.get_data(data.CROP_DIR)
randomForest = RandomForestClassifier(n_estimators=1, n_jobs=-1, random_state=90)
for i in range(min_estimators, max_estimators + 1, 5):
        randomForest.set_params(n_estimators=i)
        rf=randomForest.fit(x_train, y_train)
        train_scores.append(cross_val_score(rf, x_train, y_train, cv=3))
        #test_scores.append(randomForest.score(X_test, y_test))

fig, ax = plt.subplots(dpi = 100)
ax.set_xlabel("estimators")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs estimators for training and testing sets")
ax.plot(range(min_estimators, max_estimators + 1, 5), train_scores, label="train",
        drawstyle="steps-post")
#ax.plot(range(min_estimators, max_estimators + 1, 5), test_scores, label="test",
#        drawstyle="steps-post")
ax.legend()
plt.show()
import os
import pandas
from src.data import Data
from sklearn.model_selection import train_test_split

data=Data()
original_dir_list=os.listdir(data.ORIGINAL_DIR)
csv_list=[]
for i in original_dir_list:
    data_dir=os.path.join(data.ORIGINAL_DIR,i)
    for j in os.listdir(data_dir):
        if j.endswith('.csv'):
            file=os.path.join(data_dir,j)
            temp=pandas.read_csv(file,sep=";").values
            csv_list.append(temp)
for i in range(len(original_dir_list)):
    train, test=train_test_split(csv_list[i],train_size=0.7,test_size=0.3)
    train_dir=os.path.join(data.TRAIN_DIR,original_dir_list[i])
    test_dir=os.path.join(data.TEST_DIR,original_dir_list[i])
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    else:
        os.removedirs(train_dir)
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        os.removedirs(test_dir)
        os.mkdir(test_dir)

    





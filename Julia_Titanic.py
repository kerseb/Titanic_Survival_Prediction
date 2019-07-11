# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:27:02 2019

@author: bm52gw
"""

import pandas as pd
import numpy as np

import DataHandling

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesRegressor




train_path = 'H:/Julia/07_workspace/Sebastian/Titanic_Survival_Prediction/Data/train.csv'
test_path = 'H:/Julia/07_workspace/Sebastian/Titanic_Survival_Prediction/Data/test.csv'


def load_data(data_path, train):
    data = pd.read_csv(data_path)
#    datam = data.as_matrix()
    x_data = data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    if train:
        y_data = data['Survived']
    
    oh_enc = OneHotEncoder(sparse=False)
    label_enc = LabelEncoder()
    
    # one hot encoding of sex parameter
    sex = data['Sex']
    sex_int = label_enc.fit_transform(sex)
    sex_int = sex_int.reshape(len(sex_int),1)
    sex_oh = oh_enc.fit_transform(sex_int)
    sex_oh = pd.DataFrame(data=sex_oh, columns=['Female','Male'])
    
    x_data = pd.concat([x_data,sex_oh], axis=1)
    
    # one hot encoding of embarked parameter
    embarked = data['Embarked']
    embarked = embarked.fillna('nan')
    embarked_int = label_enc.fit_transform(embarked)
    embarked_int = embarked_int.reshape(len(embarked_int),1)
    embarked_oh = oh_enc.fit_transform(embarked_int)
    if train:
        embarked_oh = pd.DataFrame(data=embarked_oh, columns=['C','Q','S','nan'])
        embarked_oh = embarked_oh.drop(['nan'],axis=1)
    else:
        embarked_oh = pd.DataFrame(data=embarked_oh, columns=['C','Q','S'])
    
    x_data = pd.concat([x_data,embarked_oh], axis=1)
    
    # age correction
    x_data['Age'].fillna(-1, inplace=True)
    x_data['Age'] = np.ceil(x_data['Age']).astype(int)
    
    x_data['Fare'].fillna(-1, inplace=True)
    x_data['Fare'] = np.ceil(x_data['Fare']).astype(int)
    
    #final
    x_data = x_data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Female', 'Male', 'C', 'Q', 'S']]
    test_id = data['PassengerId']
    
    if train:
        return x_data, y_data
    else:
        return x_data, test_id



x_train, y_train = load_data(train_path,True)
x_test, test_id = load_data(test_path,False)


clf = ExtraTreesRegressor(n_estimators=1000, random_state = 50, max_features='auto', n_jobs=-1)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)

pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0


DataHandling.WriteResults('Data/Julia_submission.csv',test_id,pred)





    
import csv
import Classifier
import DataHandling
import numpy as np
import math

""" Data import and handling """

#train data
train_data_dir = 'Data/train.csv'
train_data = DataHandling.ImportData(train_data_dir)
train_data = DataHandling.CleanData(train_data)

n = len(train_data)
print('Number of data available: ', n)

train_data = np.array(train_data)
n_train = math.ceil(n * 0.95)

X_train = train_data[:n_train,:-1]
Y_train = train_data[:n_train,-1]

X_eval = train_data[n_train:,:-1]
Y_eval = train_data[n_train:,-1]

# test data
test_data_dir = 'Data/test.csv'
test_data = DataHandling.ImportData(test_data_dir)
test_data = DataHandling.CleanDataTest(test_data)

n_test = len(test_data)
print('Number of test data: ', n_test)

test_data = np.array(test_data)
test_id = test_data[:,0]
X_test = test_data[:,1:]


""" Generate Classifier """
print('\nDesicion Tree Classifier: \n')
DTC = Classifier.DecisionTreesClassifier(X_train,Y_train)
DTC_score = DTC.evaluate(X_eval,Y_eval)

print('\nK-Nearest-Neighbor: \n')

kNN = Classifier.KNearestNeighbor(X_train, Y_train, neighbors = [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80], weights = ['distance'], algorithm = 'auto',n_jobs=1)
kNN_score = kNN.evaluate(X_eval,Y_eval)

print('\nNeural-Network with adaptive parameter choice: \n')

layer = [1,2,3,4,5,10,20]
neurons = [1,2,3,4,5,10,15,20,30]
#testcase
#layer = [10]
#neurons = [20]

NN , err = Classifier.AdaptiveChoiceNN(X_train,Y_train,layer,neurons,epochs_final = 2000, epochs = 500,batch_size = 32, activation_fun = 'relu', loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'],rand_seed = True)
NN_score = NN.evaluate(X_eval,Y_eval)


""" Label and save test data """
Y_pred_DTC = DTC.predict(X_test)
Y_pred_DTC[Y_pred_DTC > 0.5] = 1
Y_pred_DTC[Y_pred_DTC <= 0.5] = 0

Y_pred_kNN = kNN.predict(X_test)
Y_pred_kNN[Y_pred_kNN > 0.5] = 1
Y_pred_kNN[Y_pred_kNN <= 0.5] = 0

Y_pred_NN = NN.predict(X_test)
Y_pred_NN[Y_pred_NN > 0.5] = 1
Y_pred_NN[Y_pred_NN <= 0.5] = 0

DataHandling.WriteResults('Data/DTC_submission.csv',test_id,Y_pred_DTC)
DataHandling.WriteResults('Data/kNN_submission.csv',test_id,Y_pred_kNN)
DataHandling.WriteResults('Data/NN_submission.csv',test_id,Y_pred_NN)


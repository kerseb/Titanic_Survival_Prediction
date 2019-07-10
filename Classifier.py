from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import keras.backend as K

from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score

from sklearn import tree

import os
import numpy as np
import csv


""" Neural Network and parameter evaluation methods """
class NeuralNetwork:
    'A simple feed-forward neural network class based on keras with tensorflow backend'
    
    
    def __init__(self, in_dim = 1, out_dim = 1, hidden_layer = 1, neur = 1, 
                 act = 'relu',loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'], rand_seed = False, load_path = ''):
        
        K.clear_session()
        self.model = Sequential()

        if(load_path != ''):
            self.load_network(load_path)
            return
        
        if rand_seed:
            np.random.seed(7)
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.layer = hidden_layer +2
        self.loss_fun = loss_fun
        self.optimizer = optimizer 
        self.metrics = metrics
        
        if isinstance(act, list):
            self.activation_fun = act
        else:
            self.activation_fun = []
            for i in range(self.layer-1):
                self.activation_fun.append(act)
            self.activation_fun.append('sigmoid')
            
        if isinstance(neur, list):
            self.neurons = neur
        else:
            self.neurons = []
            for i in range(self.layer-1):
                self.neurons.append(neur)
        
        self.add_layer()
        self.model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=self.metrics)
        
    def add_layer(self):        
        #initialize input layer
        self.model.add(Dense(self.neurons[0], input_dim=self.input_dim, activation=self.activation_fun[0]))
        #initialize hidden layer
        for i in range(1,self.layer -1):
            self.model.add(Dense(self.neurons[i], activation=self.activation_fun[i]))
        #initialize output layer
        self.model.add(Dense(self.output_dim, activation=self.activation_fun[-1]))
        return True
    
    def count_params(self):
        return self.model.count_params()
        #return int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))


    def fit(self,X,Y,epochs = 300, batch_size = 32, verbose = 0):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose = verbose)
        return True

    
    def evaluate(self,X,Y):
        scores = self.model.evaluate(X, Y , verbose = 0)
        print("%s: %.4f" % ("MSE", scores[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        return scores
    
    def predict(self,X):
        return self.model.predict(X)
    
    def save_network(self,path = ''):
        abs_path = os.path.abspath(path)

        try:
            os.mkdir(abs_path)
        except FileExistsError:
            pass
        
        abs_path += '/'
        
        # save network parameter
        model_par = open(abs_path + 'model_par.csv','w')
        parameter =  csv.writer(model_par)
        parameter.writerow([self.input_dim])
        parameter.writerow([self.output_dim])
        parameter.writerow([self.layer])
        parameter.writerow(self.activation_fun)   
        parameter.writerow(self.neurons)
        parameter.writerow([self.loss_fun])
        parameter.writerow([self.optimizer])
        parameter.writerow(self.metrics)   
        model_par.close()
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(abs_path + "model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(abs_path + "model.h5")
        print("Saved model to disk")
        return True
    
    def load_network(self,path = ''):
        abs_path = os.path.abspath(path)
        abs_path += '/'
        
        # load model parameter 
        if os.path.exists(abs_path + 'model_par.csv'): 
            model_par = open(abs_path + 'model_par.csv','r')
            parameter =  csv.reader(model_par, delimiter='\n')
            self.input_dim = parameter.__next__()[0]
            self.output_dim = parameter.__next__()[0]
            self.layer = parameter.__next__()[0]
            self.activation_fun = parameter.__next__()   
            self.neurons = parameter.__next__()
            self.loss_fun = parameter.__next__()
            self.optimizer = parameter.__next__()[0]
            self.metrics = parameter.__next__()
            model_par.close()      
        else:
            print('File :' + abs_path + 'model_par.csv' + ' does not exists')
            return False
        
        
        
        # load json and create model
        if os.path.exists(abs_path + 'model.json'): 
            json_file = open(abs_path + 'model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
        else:
            print('File :' + abs_path + 'model.json' + ' does not exists')
            return False
        # load weights into new model
        if os.path.exists(abs_path + 'model.h5'): 
            self.model.load_weights(abs_path + 'model.h5')
        else:
            print('File :' + abs_path + 'model.h5' + ' does not exists')
            return False
        
        # compile model
        self.model.compile(loss=self.loss_fun, optimizer=self.optimizer, metrics=self.metrics)
        
        print("Loaded model from disk")
        return True

    def get_model_param(self):
        return {'input dimension':self.input_dim,
                'output dimension':self.output_dim,
                'hidden layer':self.layer,
                'activation functions':self.activation_fun,
                'number of neurons':self.neurons,
                'loss function':self.loss_fun,
                'optimizer':self.optimizer,
                'metrics':self.metrics}


    
def AdaptiveChoiceNN(X,Y,hidden_layer,neurons,epochs_final = 2000, epochs = 500,batch_size = 32, activation_fun = 'sigmoid', loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'],rand_seed = True):
    #if optimizer is 'sgd':
    #    sgd = keras.optimizers.SGD(lr=0.0, momentum=0.0, decay=0.0, nesterov=False)
    (n,d) = np.shape(X)
    n_train = int(np.ceil(n * 2/3));
    n_test = n - n_train;
    best_non = 0;
    best_layer = 0;
    best_err = np.inf;

    for layer in hidden_layer:
        for non in neurons:
            
            #initilize NN
            NN = NeuralNetwork(d, 1, hidden_layer = layer, neur = non, act = activation_fun,loss_fun = loss_fun, optimizer = optimizer , metrics = metrics, rand_seed = rand_seed)
            
            """
            # test if number of weights > max(5 * n_train,200)
            print(layer,non)
            print(NN.count_params())
            if NN.count_params() > max(5*n_train,200):
                continue;
            """

            # train NN
            if( batch_size == -1 ):
                batch_size = n_train

            NN.fit(X[0:n_train,:],Y[0:n_train],epochs = epochs, batch_size = batch_size)

            error = 0;
            for i in range(n_train+1,n):
                error += np.power(NN.predict(X[i:i+1])[0][0] - Y[i],2)

            error /= n_test
            #print(layer,non,error)
            if error < best_err:
                best_layer = layer;
                best_non = non;
                best_err = error;
                #best_NN = NN;


    


    #print('Adaptive Choice')
    print('chosen number of layers: %i' % best_layer)
    print('chosen number of neurons per layer: %i' % best_non)
    #print("L2 - Error: ",best_err)

    #best_NN.fit(X[1:n_train,:],Y[1:n_train],epochs = epochs_final - epochs, batch_size = batch_size, verbose = 1) 

    best_NN = NeuralNetwork(d, 1, hidden_layer = best_layer, neur = best_non, act = activation_fun,loss_fun = loss_fun, optimizer = optimizer , metrics = metrics, rand_seed = rand_seed)
    best_NN.fit(X,Y,epochs = epochs_final, batch_size = batch_size)

    return best_NN , best_err
    


def CrossValidationNN(X,Y,hidden_layer,neurons,epochs = 300,batch_size = 32, activation_fun = 'relu', loss_fun = 'mean_squared_error', optimizer = 'adam' , metrics = ['accuracy'],rand_seed = True):
    # make hidden_layer and neurons iterable if they are just a number
    if not isinstance(hidden_layer, list):
        hidden_layer = [hidden_layer]
    if not isinstance(neurons, list):
        neurons = [neurons]
    
    # get input and output dimension
    try:
        input_dim = np.size(X,1)    
    except IndexError:
        input_dim = 1
    try:
        output_dim = np.size(Y,1)    
    except IndexError:
        output_dim = 1
    
    # initiate cross-validation scores and parameter list, and get cv indices
    cv_scores = []
    param = []
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(X, Y))

    # iterate over each parameter combination
    for h in hidden_layer:
        for n in neurons:
            
            #initilize NN
            NN = NeuralNetwork(input_dim, output_dim, hidden_layer = h, neur = n, act = activation_fun,loss_fun = loss_fun, optimizer = optimizer , metrics = metrics, rand_seed = rand_seed)
            
            print(NN.get_model_param())

            scores = []
            """
            scores = cross_val_score(NN, X, Y, cv=5, scoring='neg_mean_squared_error')
            """

            for j, (train_idx, val_idx) in enumerate(folds):
                print('\nFold ',j+1)
                # splitt data
                X_train_cv = X[train_idx]
                Y_train_cv = Y[train_idx]
                X_valid_cv = X[val_idx]
                Y_valid_cv = Y[val_idx]
                # train NN
                NN.fit(X_train_cv,Y_train_cv,epochs = epochs, batch_size = batch_size)
                # evaluate NN
                print(NN.evaluate(X_valid_cv,Y_valid_cv)[0])
                scores.append(NN.evaluate(X_valid_cv,Y_valid_cv)[0])
            cv_scores.append(np.array(scores).mean())
            param.append([h,n])

            del NN

    optimal_p = param[cv_scores.index(min(cv_scores))]
    optimal_h = optimal_p[0]
    optimal_n = optimal_p[1]
        
    print("Optimal number of hidden layer is: %i" % (optimal_h))
    print("Optimal number of neurons is: %s" % (optimal_n))
    return optimal_h , optimal_n, min(cv_scores) 

""" K-Nearest-Neighbor methods """

class KNearestNeighbor:
    model = KNeighborsRegressor()

    def __init__(self, X, Y, neighbors = [1,5,20], weights = 'uniform', algorithm = 'auto',n_jobs=1):
        """
        X,Y:       traindata
        neighbors: number of neighbors, if list than 5-fold cross-validation is used to get optimal value
        weights:   must be one of ['uniform', 'distance'], if list than 5-fold cross-validation is used to get optimal value
        algorithm: must be one of [‘auto', ‘ball_tree', ‘kd_tree', ‘brute']
        n_jobs:    The number of parallel jobs to run for neighbors search. If -1, then the number of jobs is set to the number of CPU cores. Doesn't affect fit method.

        """
        if isinstance(neighbors, list) or isinstance(weights, list):
            neighbors , weights, optimal_err = self.cross_validation(X,Y,neighbors,weights,algorithm,n_jobs)
            
        self.model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
        self.model.fit(X,Y)

    def evaluate(self,X,Y):
        Y_pred = self.predict(X)
        mse = mean_squared_error(Y, Y_pred)
        Y_pred[Y_pred > 0.5] = 1
        Y_pred[Y_pred <= 0.5] = 0
        acs = accuracy_score(Y,Y_pred)
        print("\n%s: %.4f" % ("MSE", mse))
        print("%s: %.2f%%" % ("Accuracy", acs*100))
        return mse,acs
    
    def predict(self,X):
        return self.model.predict(X)

    def kneighbors(X=None, n_neighbors=None, return_distance=True):
        return self.model.kneighbors(X = X, n_neighbors=n_neighbors, return_distance=return_distance)

    def cross_validation(self,X,Y,neighbors,weights,algorithm,n_jobs,cv = 5):
        if not isinstance(neighbors, list):
            neighbors = [neighbors]
        if not isinstance(weights, list):
            weights = [weights]
        
        cv_scores = []
        param = []
        for k in neighbors:
            for w in weights:
                self.model = KNeighborsRegressor(n_neighbors=k, weights=w, algorithm=algorithm, n_jobs=n_jobs)
                scores = cross_val_score(self.model, X, Y, cv=cv, scoring='neg_mean_squared_error')
                cv_scores.append(-1*scores.mean())
                param.append([k,w])
        #print(cv_scores)
        optimal_p = param[cv_scores.index(min(cv_scores))]
        optimal_k = optimal_p[0]
        optimal_weights = optimal_p[1]
        print("Optimal value for neighbors is: %i" % (optimal_k))
        print("Optimal weight method is: %s" % (optimal_weights))
        return optimal_k , optimal_weights, min(cv_scores)

    def get_params(self,deep = True):
        return self.model.get_params(deep)


""" Decision Trees """

class DecisionTreesClassifier:
    def __init__(self, X, Y, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state = 1, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False):
        """
        X,Y: traindata
        criterion : string, optional (default=”gini”); The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        splitter : string, optional (default=”best”); The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
        max_depth : int or None, optional (default=None); The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        min_samples_split : int, float, optional (default=2); The minimum number of samples required to split an internal node: 
            - If int, then consider min_samples_split as the minimum number.
            - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
        min_samples_leaf : int, float, optional (default=1); The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
            - If int, then consider min_samples_leaf as the minimum number.
            - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
        min_weight_fraction_leaf : float, optional (default=0.); The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
        max_features : int, float, string or None, optional (default=None); The number of features to consider when looking for the best split:
            - If int, then consider max_features features at each split.
            - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
            - If “auto”, then max_features=sqrt(n_features).
            - If “sqrt”, then max_features=sqrt(n_features).
            - If “log2”, then max_features=log2(n_features).
            - If None, then max_features=n_features.
        random_state : int, RandomState instance or None, optional (default=None); 
            - If int, random_state is the seed used by the random number generator; 
            - If RandomState instance, random_state is the random number generator; 
            - If None, the random number generator is the RandomState instance used by np.random.

        max_leaf_nodes : int or None, optional (default=None); Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
        min_impurity_decrease : float, optional (default=0.); A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        The weighted impurity decrease equation is the following:
            - N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)
              where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
              N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.

        min_impurity_split : float, (default=1e-7); Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
        class_weight : dict, list of dicts, “balanced” or None, default=None; Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
        presort : bool, optional (default=False); Whether to presort the data to speed up the finding of best splits in fitting. For the default settings of a decision tree on large datasets, setting this to true may slow down the training process. When using either a smaller dataset or a restricted depth, this may speed up the training.

        """
        """
        if isinstance(neighbors, list) or isinstance(weights, list):
            neighbors , weights, optimal_err = self.cross_validation(X,Y,neighbors,weights,algorithm,n_jobs)
            
        self.model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, algorithm=algorithm, n_jobs=n_jobs)
        """
        self.model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, class_weight=class_weight, presort=presort)
        self.model.fit(X,Y)

    def evaluate(self,X,Y):
        Y_pred = self.predict(X)
        mse = mean_squared_error(Y, Y_pred)
        Y_pred[Y_pred > 0.5] = 1
        Y_pred[Y_pred <= 0.5] = 0
        acs = accuracy_score(Y,Y_pred)
        print("\n%s: %.4f" % ("MSE", mse))
        print("%s: %.2f%%" % ("Accuracy", acs*100))
        return mse,acs
    
    def predict(self,X):
        return self.model.predict(X)
    """
    def cross_validation(self,X,Y,neighbors,weights,algorithm,n_jobs,cv = 5):
        if not isinstance(neighbors, list):
            neighbors = [neighbors]
        if not isinstance(weights, list):
            weights = [weights]
        
        cv_scores = []
        param = []
        for k in neighbors:
            for w in weights:
                self.model = KNeighborsRegressor(n_neighbors=k, weights=w, algorithm=algorithm, n_jobs=n_jobs)
                scores = cross_val_score(self.model, X, Y, cv=cv, scoring='neg_mean_squared_error')
                cv_scores.append(-1*scores.mean())
                param.append([k,w])
        #print(cv_scores)
        optimal_p = param[cv_scores.index(min(cv_scores))]
        optimal_k = optimal_p[0]
        optimal_weights = optimal_p[1]
        print("Optimal value for neighbors is: %i" % (optimal_k))
        print("Optimal weight method is: %s" % (optimal_weights))
        return optimal_k , optimal_weights, min(cv_scores)

    def get_params(self,deep = True):
        return self.model.get_params(deep)
    """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['figure.figsize'] = (10.0, 4.0)

import itertools

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# Cross-validation object
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('winequality-white.csv', sep=';')

dataset.shape

features = dataset.columns[0:dataset.columns.shape[0]-1]

features
target_names = dataset.groupby('quality').size().index

X = dataset[features]
Y = dataset['quality']

X = X.as_matrix()
Y = Y.as_matrix()

MyKNeighbors = KNeighborsClassifier(n_neighbors=3)

def generateTrainTest():
    myStratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=10) 
    for train_index, test_index in myStratifiedShuffleSplit.split(X,Y):
        Xtrain = X[train_index,:] 
        Xtest = X[test_index,:]
        Ytrain = Y[train_index] 
        Ytest = Y[test_index]
        return Xtrain, Xtest, Ytrain, Ytest

Xtrain, Xtest, Ytrain, Ytest = generateTrainTest()

mi_param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15], 'weights':['uniform', 'distance'], 'metric':['euclidean', 'manhattan', 'chebyshev']}

mySplits = 10
myCVStratifiedKFold = StratifiedKFold(n_splits=mySplits, shuffle=True, random_state=10)

myGridSearchCV= GridSearchCV(MyKNeighbors,mi_param_grid,cv=myCVStratifiedKFold,verbose=0, return_train_score=True)



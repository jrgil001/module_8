import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
plt.rcParams['figure.figsize'] = (10.0, 4.0)

import itertools

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# Cross-validation object
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from sklearn.decomposition import PCA
#%% Import and prepare data

#dataset = pd.read_csv('C:/Users/Manuel/Desktop/Final/winequality-white.csv', sep=';')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ['wine','Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity', 'Hue', 'OD280/OD315', 'Proline']
dataset = pd.read_csv(url,   names=names)

dataset.shape

#features = dataset.columns[0:dataset.columns.shape[0]-1]
features = dataset.columns[1:dataset.columns.shape[0]]

target_names = dataset.groupby('wine').size().index

features = np.asarray(features)
target_names = np.asarray(target_names, dtype=np.str)

X = dataset[features]
Y = dataset['wine']

X = X.as_matrix()
Y = Y.as_matrix()



#%% GridSearchCV -> CrossValidation


def generateTrainTest(random_state = int(time.time())):
    myStratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1,test_size=0.3, random_state=random_state) 
    for train_index, test_index in myStratifiedShuffleSplit.split(X,Y):
        Xtrain = X[train_index,:] 
        Xtest = X[test_index,:]
        Ytrain = Y[train_index] 
        Ytest = Y[test_index]
        return Xtrain, Xtest, Ytrain, Ytest

def only_one_param_to_plot(mi_param_grid):
    num_param_to_plot = 0
    actual_param_to_plot = 0
    for param in mi_param_grid:
        if len(mi_param_grid[param])>1:
            num_param_to_plot +=1
            actual_param_to_plot = param
    if num_param_to_plot == 1: return actual_param_to_plot
    else: return False

def validationCurves(myGridSearchCV, param_name, ylimit = [0.5, 1.01]):
    myGridSearchCVResult = myGridSearchCV.cv_results_
    train_scores_mean = myGridSearchCVResult['mean_train_score']
    train_scores_std = myGridSearchCVResult['std_train_score']
    test_scores_mean = myGridSearchCVResult['mean_test_score']
    test_scores_std = myGridSearchCVResult['std_test_score']
    

    param_range = np.array([])
    for param in myGridSearchCVResult['params']: param_range = np.append(param_range,param[param_name])
    #param_range = param_range.astype(int)
    
    plt.title("Validation Curve")
    #plt.gca().invert_xaxis()
    plt.xlabel(param_name)
    plt.ylabel("Score")
    #plt.ylim(ylimit)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    
    plt.plot(myGridSearchCV.cv_results_['params'][myGridSearchCV.best_index_][param_name],
             myGridSearchCV.cv_results_['mean_test_score'][myGridSearchCV.best_index_],
            "ro", label="Best Estimator")

    plt.legend(loc="best")
    plt.show()


def printCV(CV):
    print("Accuracy:\t %0.2f (+/- %0.2f)" % (np.mean(CV) * 100, np.std(CV) *100 * 2) + " %")

def plotCV(CV):
    plt.plot(range(len(CV)),CV,'b',label="accuracy")
    plt.plot((0,len(CV)-1),(np.mean(CV),np.mean(CV)),'r--',label="mean acurracy")
    plt.fill_between(range(len(CV)),
                     np.mean(CV) - np.std(CV), np.mean(CV) + np.std(CV),
                     alpha=0.1, color="r", label="std acurracy")
    plt.xlabel ('Splits')
    plt.ylabel ('accuracy (%)')
    plt.legend(loc="best")

    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(CV) * 100, np.std(CV) * 100 * 2) + " %")

def acurracyGridSearchCV(myGridSearchCV):
    std_best_test_ = myGridSearchCV.cv_results_['std_test_score'][myGridSearchCV.best_index_]
    print("Best Estimator")
    print("Accuracy Train: %0.2f (+/- %0.2f)" % (myGridSearchCV.cv_results_['mean_train_score'][myGridSearchCV.best_index_] * 100, myGridSearchCV.cv_results_['std_train_score'][myGridSearchCV.best_index_] * 100 * 2) + " %")
    print("Accuracy CrossValidation: %0.2f (+/- %0.2f)" % (myGridSearchCV.best_score_ * 100, std_best_test_ * 100 * 2) + " %")
    

#%% confusion_matrix (Test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#%% classification_report (Test)
def show_values(pc, fmt="%.2f", **kw):
    '''
    Heatmap with text in each cell with matplotlib's pyplot
    Source: https://stackoverflow.com/a/25074150/395857 
    By HYRY
    '''
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    Inspired by:
    - https://stackoverflow.com/a/16124677/395857 
    - https://stackoverflow.com/a/25074150/395857
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    '''
    Plot scikit-learn classification report.
    Extension based on https://stackoverflow.com/a/31689645/395857 
    '''
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        #print(v)
        plotMat.append(v)

    #print('plotMat: {0}'.format(plotMat))
    #print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)




#%% Learning Curve

def plot_learning_curve(estimator,  X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), title= 'Learning Curves'):
   
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    
    plt.show()
    

#%%

class BestModel:
    
    def __init__(self, X, Y, estimator, param_grid, cv, n_jobs = -1, target_names='', showPlots=True):
        
        #Inputs
        self.X, self.Y = X, Y
        self.estimator, self.param_grid = estimator, param_grid
        self.cv = cv
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = generateTrainTest()
        self.n_jobs = n_jobs
        self.target_names = target_names
        
        #outputs
        self.CrossValidation()        
        self.AccuracyTest()
        
        self.matrixEstimator()  
        self.classification_report()
            
        self.CrossTesting()
        
        self.printResult()
        
        if showPlots:
           self.plotGSCV()
           self.plot_matrixEstimator()
           self.plot_classification_report()
           self.plotLearningCurve()
           self.plotPCA()
        
    def CrossValidation(self):
        self.GridSearchCV = GridSearchCV(self.estimator, self.param_grid, cv=self.cv, verbose=0, return_train_score=True, n_jobs = self.n_jobs)        
        self.GridSearchCV.fit(self.Xtrain, self.Ytrain)        
               
    def plotGSCV(self):
        if only_one_param_to_plot(self.param_grid):
            param_to_plot = only_one_param_to_plot(self.param_grid)
            validationCurves(self.GridSearchCV, param_to_plot, ylimit = [0.2, 1.25])

    def AccuracyTest(self):
        bestEstimator = self.GridSearchCV.best_estimator_
        bestEstimator.fit(self.Xtrain, self.Ytrain)
        self.Ypred = bestEstimator.predict(self.Xtest)
        self.accuracyTest = accuracy_score(self.Ytest, self.Ypred, normalize=True, sample_weight=None)
        
        
    def matrixEstimator(self):
        np.set_printoptions(precision=2)
        self.matrix_Estimator = confusion_matrix(self.Ytest, self.Ypred)
        self.matrix_Estimator_normalize = self.matrix_Estimator.astype('float') / self.matrix_Estimator.sum(axis=1)[:, np.newaxis]
        
    def plot_matrixEstimator(self):    
        if len(self.target_names): 
            plt.figure()
            plot_confusion_matrix(self.matrix_Estimator, classes=self.target_names,
                                  title="Confusion matrix, without normalization")
            
            plt.figure()
            plot_confusion_matrix(self.matrix_Estimator, classes=self.target_names, normalize=True,
                                  title='Normalized confusion matrix')
          
            plt.show()
        
    def classification_report(self):
        if len(self.target_names):
            self.classification_report = classification_report(self.Ytest, self.Ypred, self.target_names)
    
    def plot_classification_report(self):
        if len(self.target_names):
            plot_classification_report(self.classification_report)
    
    def CrossTesting(self):
        self.scoreTesting = cross_val_score(self.GridSearchCV.best_estimator_,self.X,self.Y,cv=self.cv)

    def plotLearningCurve(self):
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        plot_learning_curve(self.GridSearchCV.best_estimator_, self.X, self.Y, cv=cv, n_jobs=self.n_jobs)

    def plotPCA(self):
        myPCA = PCA(n_components=2)
        X_PCA = myPCA.fit_transform(self.X) 

        myModel = self.GridSearchCV
        myModel.fit(X_PCA, Y)
        
        xx1, xx2 = np.meshgrid(np.linspace(X_PCA[:,0].min(),X_PCA[:,0].max(),100), np.linspace(X_PCA[:,1].min(),X_PCA[:,1].max(),100))
        
        Z = myModel.predict(np.c_[xx1.ravel(),xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        
        plt.figure()
        plt.contourf(xx1, xx2, Z, alpha=0.4)
        plt.scatter(X_PCA[:,0],X_PCA[:,1],s=100,c=Y)
        plt.show()

        #print(myPCA.explained_variance_ratio_)
        
    def printResult(self):
            print("Accuracy Train:\t\t\t %0.2f (+/- %0.2f)" % (self.GridSearchCV.cv_results_['mean_train_score'][self.GridSearchCV.best_index_] * 100, self.GridSearchCV.cv_results_['std_train_score'][self.GridSearchCV.best_index_] * 100 * 2) + " %")
            print("Accuracy CrossValidation:\t %0.2f (+/- %0.2f)" % (self.GridSearchCV.best_score_ * 100, self.GridSearchCV.cv_results_['std_test_score'][self.GridSearchCV.best_index_] * 100 * 2) + " %")
            print("Accuracy Test:\t\t\t " + str(round(self.accuracyTest*100,2)) + " %")
            print("Accuracy CrossTesting:\t\t %0.2f (+/- %0.2f)" % (np.mean(self.scoreTesting) * 100, np.std(self.scoreTesting) *100 * 2) + " %")
               



#%%
myCVStratifiedKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=142)
        
MyKNeighbors = KNeighborsClassifier()
my_param_grid = {'n_neighbors': [19,17,15,13,11,9,7,5,3,1], 'weights':['uniform']}
MyBestKNN = BestModel(X=X, Y=Y, estimator = MyKNeighbors, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )


MySCV = SVC()
my_param_grid = {'kernel':['linear'],'gamma': np.logspace(-6, -1, 5)}
MyBestSVC = BestModel(X=X, Y=Y, estimator = MySCV, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )

MyTree = DecisionTreeClassifier()
my_param_grid = {'max_depth': [1,3,5,7,9,11]}
MyBestTree = BestModel(X=X, Y=Y, estimator = MyTree, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )

MyGaussianNB = GaussianNB()
my_param_grid = {}
MyBestGaussianNB = BestModel(X=X, Y=Y, estimator = MyGaussianNB, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )

MyMLPClassifier = MLPClassifier()
my_param_grid = {'hidden_layer_sizes': [1, 100, 200, 300, 400, 500, 1000, 3000]}
MyBestMLPClassifier = BestModel(X=X, Y=Y, estimator = MyMLPClassifier, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )

MyRandomForest = RandomForestClassifier()
my_param_grid = {'max_depth': [3, 5, 7, 9]}
MyBestRandomForest = BestModel(X=X, Y=Y, estimator = MyRandomForest, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )


MyAdaBoost = AdaBoostClassifier()
my_param_grid = {}
MyBestAdaBoost = BestModel(X=X, Y=Y, estimator = MyAdaBoost, param_grid=my_param_grid, cv=myCVStratifiedKFold, target_names = target_names )

#%% Algorithm Comparison

results = []
results.append(MyBestKNN.scoreTesting)
results.append(MyBestSVC.scoreTesting)
results.append(MyBestTree.scoreTesting)
results.append(MyBestGaussianNB.scoreTesting)
results.append(MyBestMLPClassifier.scoreTesting)
results.append(MyBestRandomForest.scoreTesting)
results.append(MyBestAdaBoost.scoreTesting)
results

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
namesAlgorithm = ['KNN','SCV','Tree','Gaussian','MLP','RamdomForest','AdaBoost']
ax.set_xticklabels(namesAlgorithm)
plt.ylabel('Accuracy')
plt.ylim(ymax = 1)
plt.show()
   

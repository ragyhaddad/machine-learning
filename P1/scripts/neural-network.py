#!/bin/bash/Python
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import tree
from sklearn.metrics import mean_squared_error
import numpy as np 
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import graphviz 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
## Globals
df = 0
X = 0
y = 0
def loadDataI():
    global df,X,y
    df = pd.read_csv('../datasets/network-intrusions/pcap-corrected.csv',header=None)
    le = preprocessing.LabelEncoder()
    cols_to_drop = [1,2,3]
    for x in cols_to_drop:
        le.fit(df.iloc[:,x])
        df.iloc[:,x] = le.transform(df.iloc[:,x])
    ## FEATURES AND LABEL SELECTION
    X = df.iloc[:,0:39].values
    y = df.iloc[:,42].values
def loadDataII():
    global df,X,y
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:,1:len(df.columns)-1].values
    y = df_label.iloc[:,1].values
##Main Neural Network I
def nnI(trainSize=0.8):
    global df,X,y
    loadDataI()
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=True)
    clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,100), random_state=1,max_iter=80,activation='logistic')
    clf = clf.fit(X_train,y_train)
    print 'Accuracy on Test Data:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Data:'
    print clf.score(X_train,y_train)
##Main Neural Network II
def nnII(trainSize=0.8):
    global df,X,y
    loadDataII()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=False)
    ##Load Image data
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(35, 1000),max_iter=100, random_state=1,activation='logistic')
    clf = clf.fit(X_train,y_train)
    print 'RESULTS FOR DATASET II'
    ## PRUNING FUNCTION
    print 'Accuracy on Test Set:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Set:'
    print clf.score(X_train,y_train)
## Main plotting function
def plotter(x,y_train,y_test,xlabel,outputname='graph-output.pdf'):
    plt.xlabel(xlabel)
    plt.ylabel("Score")
    plt.grid()
    plt.plot(x, y_train, 'o-', color="r",
             label="Training score")
    plt.plot(x, y_test, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    output = '../graphs/' + str(outputname)
    plt.savefig(output, dpi=199)
    plt.show()
## Iterations I
def testIterationsI():
    global df,X,y
    itterations = []
    train_scores = []
    test_scores = []
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-0.8),train_size=(0.8),shuffle=False)
    for x in np.arange(10,400,70):
        print x
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 100),max_iter=x, random_state=1,activation='logistic')
        clf = clf.fit(X_train,y_train)
        itterations.append(x)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
    plotter(itterations,train_scores,test_scores,'Iterations')
## Iterations II
def testIterationsII():
    global df,X,y
    itterations = []
    train_scores = []
    test_scores = []
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-0.8),train_size=(0.8),shuffle=False)
    for x in np.arange(10,400,70):
        print x
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 4000),max_iter=x, random_state=1,activation='logistic')
        clf = clf.fit(X_train,y_train)
        itterations.append(x)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
    plotter(itterations,train_scores,test_scores,'Iterations')
## Compare hidden layers/neurons
def plotHiddenLayerI():
    global df,X,y
    neurons = []
    train_scores = []
    test_scores = []
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(0.2),train_size=(0.8),shuffle=True)
    for x in np.arange(1,35,5):
        print x
        clf = MLPClassifier(solver='adam', alpha=1e-5,
                        hidden_layer_sizes=(16,x), random_state=1,max_iter=200,activation='logistic')
        clf = clf.fit(X_train,y_train)
        test_scores.append(clf.score(X_test,y_test))  
        train_scores.append(clf.score(X_train,y_train))
        neurons.append(x)
        print clf.score(X_test,y_test)
    plotter(neurons,train_scores,test_scores,'Number of Layers')
## Compare hidden layers/neurons - Note: reverse x,y coord to test for neurons per layer
def plotHiddenLayerII():
    global df,X,y
    neurons = []
    train_scores = []
    test_scores = []
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(0.2),train_size=(0.8),shuffle=True)
    for x in np.arange(1,1200,200):
        print x
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(35,x), random_state=1,max_iter=80,activation='logistic')
        clf = clf.fit(X_train,y_train)
        test_scores.append(clf.score(X_test,y_test))  
        train_scores.append(clf.score(X_train,y_train))
        neurons.append(x)
        print clf.score(X_test,y_test)
    plotter(neurons,train_scores,test_scores,'Number of Layers')


if __name__ == '__main__':
    print 'Dataset I'
    nnI()
    print 'Dataset II'
    nnII()


    


#!/bin/bash/Python
from sklearn.ensemble import AdaBoostClassifier
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
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
## Globals
df = 0
X = 0
y = 0

## Load Dataset I
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
## Load Dataset II
def loadDataII():
    global df,X,y
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:,1:len(df.columns)-1].values
    y = df_label.iloc[:,1].values

## Boosting I
def boostingI(trainSize=0.8):
    global df,X,y
    loadDataI()
        ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=True)
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=30, learning_rate=0.8, random_state=None)
    clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=20),
    n_estimators=50,
    learning_rate=0.5)
    clf = clf.fit(X_train,y_train)
    print 'Accuracy on Test Data:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Data:'
    print clf.score(X_train,y_train)
## Boosting I
def boostingII(trainSize=0.8):
    global df,X,y
    loadDataII()
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=True)
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=7, learning_rate=0.2, random_state=None)
    clf = clf.fit(X_train,y_train)
    print 'Accuracy on Test Data:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Data:'
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
## Plotting for varrying Estimators
def testEstimatorsI():
    global df,X,y
    loadDataI()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(1,60,10):
        print x
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=x,learning_rate=1)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'Number of Estimators',outputname='boosting-est-1.pdf')
## Plotting for Varrying Estimators
def testEstimatorsII():
    global df,X,y
    loadDataII()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(1,100,20):
        print x
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=x,learning_rate=1)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'Number of Estimators',outputname='boosting-est-2.pdf')
## Plotting for varrying Estimators
def testLRI():
    global df,X,y
    loadDataI()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(0.1,1.5,0.2):
        print x
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=11,learning_rate=x)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'Learning Rate',outputname='boosting-lr-1.pdf')
## Plotting for Varrying Estimators
def testLRII():
    global df,X,y
    loadDataII()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(0.1,1.5,0.2):
        print x
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20),n_estimators=30,learning_rate=x)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'Learning Rate',outputname='boosting-lr-2.pdf')


if __name__ == '__main__':
    print 'Dataset I'
    boostingI()
    print 'Dataset II'
    boostingII()


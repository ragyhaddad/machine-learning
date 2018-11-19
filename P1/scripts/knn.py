#!/bin/bash/Python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import numpy as np 
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
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
    X = df.iloc[:400,1:len(df.columns)-1].values
    y = df_label.iloc[:400,1].values

## Boosting I
def knnI(trainSize=0.8):
    global df,X,y
    loadDataI()
        ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=True)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf = clf.fit(X_train,y_train)
    print 'Accuracy on Test Data:'
    #   print trainSize, ',' , clf.score(X_test,y_test), ',',clf.score(X_train,y_train)
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Data:'
    print clf.score(X_train,y_train)
## Boosting I
def knnII(trainSize=0.8):
    global df,X,y
    loadDataII()
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-trainSize),train_size=(trainSize),shuffle=True)
    clf = KNeighborsClassifier(n_neighbors=3)
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
    # plt.show()
## Plotting for varrying Estimators
def testNeighborsI():
    global df,X,y
    loadDataI()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(3,60,10):
        print x
        clf = KNeighborsClassifier(n_neighbors=x)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'K Neighbors',outputname='knn-est-1.pdf')
## Plotting for Varrying Estimators
def testNeighborsII():
    global df,X,y
    loadDataII()
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    xaxis = []
    for x in np.arange(3,60,10):
        print x
        clf = KNeighborsClassifier(n_neighbors=x)
        clf = clf.fit(X_train,y_train)
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        print clf.score(X_test,y_test)
        xaxis.append(x)
    plotter(xaxis,train_scores,test_scores,'K Neighbors',outputname='knn-est-2.pdf')
def testSizeI():
    global df,X,y
    loadDataI()
    train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=3), X, y, train_sizes=np.arange(0.1,1,0.1), cv=8)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
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
             label="Test score")

    plt.legend(loc="best")
    plt.savefig('../graphs/knn-sizes1.png', dpi=199)
    plt.show() 
def testSizeII():
    global df,X,y
    loadDataII()
    train_sizes, train_scores, test_scores = learning_curve(
    KNeighborsClassifier(n_neighbors=3), X, y, train_sizes=np.arange(0.1,1,0.1), cv=8)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
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
             label="Test score")

    plt.legend(loc="best")
    plt.savefig('../graphs/knn-sizes2.png', dpi=199)
    plt.show() 
if __name__ == '__main__':
    print 'Dataset I:'
    knnI()
    print 'Dataset II:'
    knnII()

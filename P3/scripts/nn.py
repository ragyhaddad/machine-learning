#!/bin/bash/python 
import data as data_handler 
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np 
from sklearn import mixture 
import clustering as cluster 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report

def nn_1(df,X,y,components=2):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    pca = PCA(n_components=components)
    pca_fit = pca.fit(X_train)
    reduced_training = pca_fit.transform(X_train)
    reduced_testing = pca_fit.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(20,20),activation='relu',max_iter=500)
    model = model.fit(reduced_training,y_train) 
    train_score = model.score(reduced_training,y_train)
    test_score = model.score(reduced_testing,y_test)
    return train_score,test_score

def nn_2(df,X,y,components=2):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    pca = PCA(n_components=components)
    pca_fit = pca.fit(X_train)
    reduced_training = pca_fit.transform(X_train)
    reduced_testing = pca_fit.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(20,20),activation='relu',max_iter=1000)
    model = model.fit(reduced_training,y_train) 
    train_score = model.score(reduced_training,y_train)
    test_score = model.score(reduced_testing,y_test)
    return train_score,test_score

def plot_accuracy_components_1(df,X,y):
    test_scores = []
    train_scores = []
    for x in range(1,30,1):
        train_score,test_score = nn_1(df,X,y,x) 
        print train_score 
        print test_score
        train_scores.append(train_score)
        test_scores.append(test_score)
    print test_scores
    plt.grid(linestyle='dotted')
    plt.ylabel('Accuracy')
    plt.xlabel('No Components')
    plt.plot(np.arange(1,30,1),test_scores,'-o',color='orange',label='Test Score')
    plt.plot(np.arange(1,30,1),train_scores,'-o',color='blue',label='Train Score')
    plt.legend(loc='Best')
    plt.show() 

def plot_accuracy_components_2(df,X,y):
    test_scores = []
    train_scores = []
    for x in range(1,50,1):
        train_score,test_score = nn_2(df,X,y,x) 
        print train_score 
        print test_score
        train_scores.append(train_score)
        test_scores.append(test_score)
    print test_scores
    plt.grid(linestyle='dotted')
    plt.ylabel('Accuracy')
    plt.xlabel('No Components')
    plt.plot(np.arange(1,50,1),test_scores,'-o',color='orange',label='Test Score')
    plt.plot(np.arange(1,50,1),train_scores,'-o',color='blue',label='Train Score')
    plt.legend(loc='best')
    plt.show() 

def classification_report_1():
    df,X,y = data_handler.loadDataI()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    pca = PCA(n_components=3)
    pca_fit = pca.fit(X_train)
    reduced_training = pca_fit.transform(X_train)
    reduced_testing = pca_fit.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(20,20),activation='relu',max_iter=1000)
    model = model.fit(reduced_training,y_train) 
    y_pred = model.predict(reduced_testing)
    print classification_report(y_pred,y_test)

def classification_report_2():
    df,X,y = data_handler.loadDataIII()
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=False)
    pca = PCA(n_components=37)
    pca_fit = pca.fit(X_train)
    reduced_training = pca_fit.transform(X_train)
    reduced_testing = pca_fit.transform(X_test)
    model = MLPClassifier(hidden_layer_sizes=(20,20),activation='relu',max_iter=1000)
    model = model.fit(reduced_training,y_train) 
    y_pred = model.predict(reduced_testing)
    print classification_report(y_pred,y_test)







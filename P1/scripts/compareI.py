#!/bin/bash/Python
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import sys
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

## Globals
df = 0
X = 0
y = 0
df = pd.read_csv('../datasets/network-intrusions/pcap-corrected.csv',header=None)
le = preprocessing.LabelEncoder()
cols_to_drop = [1,2,3]
for x in cols_to_drop:
    le.fit(df.iloc[:,x])
    df.iloc[:,x] = le.transform(df.iloc[:,x])


## FEATURES AND LABEL SELECTION
X = df.iloc[:,0:39].values
y = df.iloc[:,42].values
print y
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,shuffle=True)
dt = DecisionTreeClassifier(criterion='entropy',max_depth=30)
boosting = AdaBoostClassifier(base_estimator=None, n_estimators=30, learning_rate=0.8, random_state=None)
knn = KNeighborsClassifier(n_neighbors=3)
nn =  MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,100), random_state=1,max_iter=80,activation='logistic')
svm = svm.LinearSVC()

models = [dt,boosting,knn,nn,svm]

clf = 0
for model in models:
    clf = model
    start = time.clock()
    clf = model.fit(X_train,y_train) 
    time_taken =  time.clock() - start
    y_pred = clf.predict(X_test)   
    print(classification_report(y_test, y_pred))
    print time_taken,',',clf.score(X_test,y_test),',',clf.score(X_train,y_train)




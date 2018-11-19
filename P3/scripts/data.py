#!/bin/bash/env python
import pandas as pd  
from sklearn import preprocessing

## Load Dataset I
def loadDataI():
    global df,X,y
    df = pd.read_csv('../datasets/network-intrusions/pcap-corrected-min.csv',header=None)
    le = preprocessing.LabelEncoder()
    cols_to_drop = [1,2,3,42]
    for x in cols_to_drop:
        le.fit(df.iloc[:,x])
        df.iloc[:,x] = le.transform(df.iloc[:,x])
    ## FEATURES AND LABEL SELECTION
    X = df.iloc[:,0:39].values
    y = df.iloc[:,42].values
    
    return df,X,y
## Load Dataset II
def loadDataII():
    global df,X,y
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:30,1:20000].values
    y = df_label.iloc[:30,1].values
    return df,X,y

## Load Dataset II
def loadDataIII():
    global df,X,y
    df = pd.read_csv('../datasets/letters/digits.csv')
    X = df.iloc[:,1:len(df.columns)-2].values
    y = df.iloc[:,len(df.columns)-1].values
    return df,X,y
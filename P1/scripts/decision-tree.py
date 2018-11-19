#!/bin/bash/Python
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import numpy as np 
import sys
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import graphviz 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree._tree import TREE_LEAF
from sklearn.preprocessing import OneHotEncoder
# import seaborn as sns; sns.set_style('whitegrid')
from sklearn.model_selection import learning_curve


## Globals
train_sizes = []
total_nodes = []
accuracy_1_test = []
accuracy_2_test = []
accuracy_1_train = []
accuracy_2_train = []
post_prune_count = 0
mean_error_1 = []

## DATASET I MAIN FUNCTION
def runDecisionTreeI(plotTree=False,trainSize=0.3,testSize=0.3,pruneTree=False, pruningThreshold=0,maxDepth=50,runComparison=False):
    global train_sizes, accuracy_1_test,accuracy_1_train,total_nodes,mean_error_1
    ## STORE CATEGORICAL COLUMNS
    cols_to_drop = [1,2,3]
    all_features = [] 
    
    ## PRUNING FUNCTION
    def prune_index(inner_tree, index, threshold):
        global post_prune_count
        if inner_tree.value[index].min() < threshold:
            # turn node into a leaf by "unlinking" its children
            print inner_tree.impurity.max()
            inner_tree.children_left[index] = TREE_LEAF
            inner_tree.children_right[index] = TREE_LEAF
        # if there are shildren, visit them as well
        if inner_tree.children_left[index] != TREE_LEAF:
            prune_index(inner_tree, inner_tree.children_left[index], threshold)
            prune_index(inner_tree, inner_tree.children_right[index], threshold)
    
    ## LOAD FEATURE NAMES FROM FILE
    features_file = open('../datasets/network-intrusions/pcap-features-all.txt',"r")
    for x in features_file.readlines():
        all_features.append(x.rstrip())
    train_features = all_features[0:39]

    ## LOAD DATASET INTO DATAFRAME
    df = pd.read_csv('../datasets/network-intrusions/pcap-corrected.csv',header=None)
    df.columns = all_features

    ## CHANGE CATEGORICAL DATA TO INTEGER TYPE LABELS USING OneHotEncoder
    le = preprocessing.LabelEncoder()
    cols_to_drop = [1,2,3]
    for x in cols_to_drop:
        le.fit(df.iloc[:,x])
        df.iloc[:,x] = le.transform(df.iloc[:,x])
    
    ## FEATURES AND LABEL SELECTION
    X = df.iloc[:,0:39].values
    y = df.iloc[:,42].values

    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=testSize,shuffle=True)
    ## FIT DATA TO MODEL
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=maxDepth)
    clf = clf.fit(X_train, y_train)
    ## PRUNE NODES THAT HAVE MINIMUM CLASS COUNT 500
    if pruneTree == True:
        prune_index(clf.tree_,0,pruningThreshold)
    ## COUNT NODES AFTER PRUNING
    count_prunes = 0
    for x in range (len(clf.tree_.value)):
        if clf.tree_.value[x].min() < pruningThreshold:
            count_prunes = count_prunes + 1      
    ## PREDICT THE VALUES OF THE TESTING SET
    predictions = clf.predict(X_test)
    ## COUNT MISPREDICTIONS
    count = 0
    for x,z in zip(y_test,predictions):
        if x != z :
            count = count + 1
    print 'RESULTS FOR DATASET I'
    print '----------------------------------------'
    print 'Test Size:'
    print len(X_test)
    print 'Train Size:'
    print len(X_train)
    print 'Accuracy on Test Data:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Data:'
    print clf.score(X_train,y_train)
    print 'Mis-Classified:'
    print str(count) + ' Out of ' + str(len(y_test))
    print 'Number of Nodes:'
    print len(clf.tree_.value)
    if pruneTree == True:
        print 'Number of Nodes After Pruning:'
        print len(clf.tree_.value) - count_prunes
    print '----------------------------------------'
    ## DRAW A DECISION TREE GRAPH
    if plotTree == True:
        dot_data = tree.export_graphviz(clf,feature_names=train_features,class_names=['normal','attack'],filled=True, rounded=True, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('test-v2')
    
    ## APPEND RESULTS TO GLOBALS
    train_sizes.append(trainSize)
    total_nodes.append(len(clf.tree_.value) - count_prunes)
    accuracy_1_test.append(clf.score(X_test,y_test))
    accuracy_1_train.append(clf.score(X_train,y_train))


# ## MAIN FUNCTION FOR DATASET II
def runDecisionTreeII(maxDepth=9999,testSize=0.3,trainSize=0.8):
    global train_sizes,accuracy_2_test,accuracy_2_train
    
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:,1:len(df.columns)-1].values
    y = df_label.iloc[:,1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=1-trainSize,train_size=trainSize,shuffle=False)
    ##Load Image data
    clf = DecisionTreeClassifier(max_depth=maxDepth)
    clf = clf.fit(X_train,y_train)
    print 'RESULTS FOR DATASET II'
    print '----------------------------------------'
    ## PRUNING FUNCTION
    print 'Accuracy on Test Set:'
    print clf.score(X_test,y_test)
    print 'Accuracy on Train Set:'
    print clf.score(X_train,y_train)
    print 'Number of Nodes:'
    print len(clf.tree_.value)
    accuracy_2_test.append(clf.score(X_test,y_test))
    accuracy_2_train.append(clf.score(X_train,y_train))
    total_nodes.append(len(clf.tree_.value))
    train_sizes.append(trainSize)
    y_pred = clf.predict(X_test)
    print 'Train Size:'
    print trainSize
    print '----------------------------------------'


## Compare Different Tree Sizes By pruning
def plotComparisonI():
    global post_prune_count
    ## LOAD DATASET INTO DATAFRAME
    df = pd.read_csv('../datasets/network-intrusions/pcap-corrected.csv',header=None)
    ## CHANGE CATEGORICAL DATA TO INTEGER TYPE LABELS USING OneHotEncoder
    le = preprocessing.LabelEncoder()
    cols_to_drop = [1,2,3]
    for x in cols_to_drop:
        le.fit(df.iloc[:,x])
        df.iloc[:,x] = le.transform(df.iloc[:,x])
    train_scores = []
    test_scores = []
    max_depth = []
    ## FEATURES AND LABEL SELECTION
    X = df.iloc[:,0:39].values
    y = df.iloc[:,42].values
    ## TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    for x in np.arange(1,200,40):
        print x
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        max_depth.append(x)   
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))

    train_sizes = max_depth
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    plt.savefig('../graphs/ds-1-prune.png', dpi=199)
    plt.show()


## Compare Different number of nodes based on max nodes due to multiclassification
def plotComparisonII():
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:,1:len(df.columns)-1].values
    y = df_label.iloc[:,1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,train_size=0.8,shuffle=True)
    train_scores = []
    test_scores = []
    max_depth = []
    for x in np.arange(10,200,40):
        print x
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        max_depth.append(x)   
        train_scores.append(clf.score(X_train,y_train))
        test_scores.append(clf.score(X_test,y_test))
        train_sizes = max_depth
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g",
             label="Test score")
    plt.legend(loc="best")
    plt.savefig('../graphs/ds-2-prune.png', dpi=199)
    plt.show()
## Compare Different Test Sizes
def plotTestSizeComparisonI():
    for x in np.arange(0.1,1,0.1):
        runDecisionTreeI(trainSize=x,plotTree=False,pruneTree=False,pruningThreshold=0)
    plt.plot(train_sizes,accuracy_1_test,linewidth=3,color='#ff5c21',label='Test Data')
    plt.plot(train_sizes,accuracy_1_train,linewidth=3,color='#232fba',label='Train Data')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.title('Train Size VS Accuracy')
    plt.legend()
    plt.savefig('../graphs/dt-1-ts.png', dpi=199)
    plt.show()

# Compare Different Test sizes
def plotTestSizeComparisonII():
    df = pd.read_csv('../datasets/expression/data.csv')
    df_label = pd.read_csv('../datasets/expression/labels.csv')
    X = df.iloc[:,1:len(df.columns)-1].values
    y = df_label.iloc[:,1].values
    train_sizes, train_scores, test_scores = learning_curve(
    DecisionTreeClassifier(criterion='entropy',max_depth=99999), X, y, train_sizes=np.arange(0.1,1,0.1), cv=8)

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
    plt.savefig('../graphs/ds-2-ts.png', dpi=199)
    plt.show() 

if __name__ == '__main__':
    runDecisionTreeI()
    runDecisionTreeII()




    




    
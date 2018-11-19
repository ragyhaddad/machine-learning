#!/bin/bash/python
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

def plot_clusters(df,X,y,components=5):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    pca = PCA(n_components=components)
    pca_model = pca.fit(X)
    reduced_train = pca.transform(X)
    kmeans_model = KMeans(n_clusters=5, random_state=100).fit(reduced_train)
    cluster_centers =  kmeans_model.cluster_centers_
    plt.grid(linestyle='dotted')
    plt.scatter(reduced_train[:,0],reduced_train[:,1],c=y,alpha=0.6,s=12)
    plt.scatter(cluster_centers[:,0],cluster_centers[:,1],color='red')
    plt.show()



def cluster_feature_nn(df,X,y,clusters=2):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)
    kmeans_model = KMeans(n_clusters=clusters, random_state=100).fit(X_train)
    features_train = kmeans_model.transform(X_train)
    features_test = kmeans_model.transform(X_test)
    nn_model = MLPClassifier(hidden_layer_sizes=(20,20),activation='relu',max_iter=700) 
    nn_model.fit(features_train,y_train) 
    train_score = nn_model.score(features_test,y_test)
    test_score = nn_model.score(features_train,y_train)
    return train_score,test_score 
    
def plot_kmeans_nn(df,X,y):
    test_scores = []
    train_scores = []
    for x in range(1,20,1):
        train_score,test_score = cluster_feature_nn(df,X,y,clusters=x)
        test_scores.append(test_score)
        train_scores.append(train_score) 
        print test_score
    plt.grid(linestyle='dotted')
    plt.ylabel('Accuracy')
    plt.xlabel('No Clusters')
    plt.plot(np.arange(1,20,1),test_scores,'-o',color='orange',label='Test Score')
    plt.plot(np.arange(1,20,1),train_scores,'-o',color='blue',label='Train Score')
    plt.legend(loc='Best')
    plt.show() 

df,X,y = data_handler.loadDataI() 
plot_kmeans_nn(df,X,y)


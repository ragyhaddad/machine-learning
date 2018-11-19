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


def cum_var_I():
    df,X,y = data_handler.loadDataI() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=total_attributes)
    pca.fit_transform(X) 

    cum_var =  pca.explained_variance_ratio_.cumsum()
    plt.title('Cumulative Variance VS Number of Components')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(0,total_attributes,1),cum_var,'-o',linewidth=3)
    plt.xlabel('Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(min(np.arange(0,total_attributes,1)), max(np.arange(0,total_attributes,1))+1, 1.0))
    plt.show()
def cum_var_II():
    df,X,y = data_handler.loadDataIII() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=total_attributes)
    pca.fit_transform(X) 

    cum_var =  pca.explained_variance_ratio_.cumsum()
    plt.title('Cumulative Variance VS Number of Components')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(0,total_attributes,1),cum_var,'-o',linewidth=3)
    plt.xlabel('Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(np.arange(min(np.arange(0,total_attributes,1)), max(np.arange(0,total_attributes,1))+1, 1.0))
    plt.show()

def pca_reconstruct(X,df):
    scores = []
    for x in range(len(df.columns)-4):
        pca = PCA(n_components=x)
        transformed = pca.fit_transform(X)
        reconstruction = pca.inverse_transform(transformed)
        mse = (np.square(X - reconstruction)).mean(axis=None)
        print mse 
        scores.append(mse) 
    plt.title('Cumulative Variance VS Number of Components')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(0,len(df.columns)-4,1),scores,'-o',linewidth=3)
    plt.xlabel('Component')
    plt.ylabel('Mean Squared Error')
    plt.savefig('../graphs/mse-2.png',dpi=299)
    plt.show()
    


#Kmeans with PCA 1
def cluster_pca_1():
    df,X,y = data_handler.loadDataI() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X) 
    cluster.sil_score(reduced_data)  
#Kmeans with PCA 2
def cluster_pca_2():
    df,X,y = data_handler.loadDataIII() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=37)
    reduced_data = pca.fit_transform(X) 
    cluster.sil_score(reduced_data)  

def cluster_pca_aic_1():
    df,X,y = data_handler.loadDataI() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X) 
    cluster.gmm(reduced_data)  

def cluster_pca_aic_2():
    df,X,y = data_handler.loadDataIII() 
    total_attributes = len(df.columns)-4
    pca = PCA(n_components=37)
    reduced_data = pca.fit_transform(X) 
    cluster.gmm(reduced_data) 


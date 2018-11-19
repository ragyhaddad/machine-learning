#!/bin/bash/python
import data as data_handler 
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np 
from sklearn import mixture
## Calculate Silhoutte Scores for for Kmeans
def sil_score(X):
    sil_scores = []
    for x in range(2,32,1):
        kmeans = KMeans(n_clusters=x, random_state=100).fit(X)
        sil = silhouette_score(X,kmeans.labels_) 
        print sil
        sil_scores.append(sil) 

    plt.title('Silhouette Score VS Number of Clusters')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(2,32,1),sil_scores,'-o') 
    # plt.savefig('../graphs/sil_score-3.png',dpi=299)
    plt.show()

def cluster_sse(X):
    scores = []
    for x in range(2,10,1):
        kmeans = KMeans(n_clusters=x, random_state=100).fit(X)
        score = kmeans.inertia_
        print score
        scores.append(score) 
    plt.title('SSE VS Number of Clusters')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(2,10,1),scores,'-o') 
    plt.savefig('../graphs/kmeans-sse-1.png',dpi=299)
    plt.show()

def gmm(X):
    scores = []
    for x in range(2,22,1):
        clf = mixture.GaussianMixture(n_components=x, covariance_type='full').fit(X)
        aic = clf.aic(X) 
        print aic
        scores.append((aic))
    plt.title('AIC VS Number of Clusters')
    plt.ylabel('AIC')
    plt.xlabel('K')
    plt.grid(linestyle='dotted')
    plt.plot(np.arange(2,22,1),scores,'-o') 
    plt.savefig('../graphs/aic-2.png',dpi=299)
    plt.show()

if __name__ == '__main__':
    #GMMs
    print 'K-Means Dataset I'
    df,X,y = data_handler.loadDataI() 
    sil_score(X) 
    print 'K-Means Dataset II' 
    df,X,y = data_handler.loadDataIII() 
    sil_score(X)  
    #GMMs
    print 'GMM Dataset I'
    df,X,y = data_handler.loadDataI() 
    gmm(X) 
    print 'GMM Dataset II' 
    df,X,y = data_handler.loadDataIII() 
    gmm(X)    

    











#!/bin/bash/python
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import itertools, sys
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Ignoring overflow warnings
np.warnings.filterwarnings('ignore')

#Globals
df = 0
X = 0
y = 0
nn = None
# Load Data
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
    y = y.reshape(-1,1) # Reshape for matrix dot product

# Neural Network Class
class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 39
        self.outputSize = 1
        self.hiddenSize = 20
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (39x20) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (39x1) weight matrix from hidden to output layer
    def forward(self, X):
        #forward propagation through our network
        self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z) # activation function
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3) # final activation function
        return o
    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error
        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")
    def predict(self,test_set):
        return self.forward(test_set)
    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)
    # alternative activation function
    def ReLU(self,x):
        return np.maximum(0.0, x)
    # derivation of relu
    def ReLU_derivation(self,x):
        if x <= 0:
            return 0
        else:
            return 1

# Random Hill Climbing
""" Random Hill Climbing is done in this portion
    First we Instantiate a Neural Network with Random Weights
    Then we calculate our Loss Function. We keep trying Random
    Initial Weights and keep the weights of the Minimum Loss we gathered"""
def randomHillSearch(no_resets=100):
    temp_loss = 1
    temp_W1 = []
    temp_W2 = []
    temp_converge = None
    for i in xrange(no_resets):
        percent = int((float(i)/no_resets)*100)
        sys.stdout.write("\r%d%%" % percent)
        sys.stdout.flush()
        nn = NeuralNetwork()
        # Minimize Loss Function based on Starting weights
        for x in range(100):
            nn.train(X_train,y_train)
            if np.mean(np.square(y_train - nn.forward(X_train))) != temp_converge:
                temp_converge = np.mean(np.square(y_train - nn.forward(X_train)))
            else:
                break
        loss = np.mean(np.square(y_train - nn.forward(X_train)))
        if loss < temp_loss:
            # print 'Best Updated Loss: ', loss
            temp_loss = loss
            temp_W1 = nn.W1 
            temp_W2 = nn.W2
    """ Set the Neural Network weights to our best weights
        (Least Loss) """
    # Set Weights
    nn.W1 = temp_W1
    nn.W2 = temp_W2
    sys.stdout.write("\r%d%%" % 100)
    sys.stdout.flush()
    sys.stdout.write('\n')
    return nn

""" Simulated Annealing Algo for optimizing Neural Network Weights """
def simulatedAnnealing(k=10,temperature=1.0):
    temp_w = 0
    temp_w_2 = 0
    random_weight_1 = 0
    random_weight_2 = 0
    Temperature = temperature
    alpha = 0.9
    Temperature_min = 0.000001
    nn = NeuralNetwork()
    loss = np.mean(np.square(y_train - nn.forward(X_train)))
    while Temperature > Temperature_min:
        percent = ((temperature - Temperature)/temperature) * 100
        sys.stdout.write("\r%f%%" % percent)
        sys.stdout.flush()
        i = 1 
        while i <= k:
            random_weight_1 = random.randint(0,len(nn.W1)-1)
            random_weight_2 = random.randint(0,len(nn.W2)-1)
            index = random.randint(0,19)
            temp_w = nn.W1[random_weight_1][index]
            nn.W1[random_weight_1][index] = random.uniform(-1,1)
            new_loss = np.mean(np.square(y_train - nn.forward(X_train)))
            if new_loss < loss:
                loss = new_loss
            else:
                nn.W1[random_weight_1][index] = temp_w
            i += 1
        Temperature = Temperature * alpha
    sys.stdout.write("\r%f%%" % 100.0)
    sys.stdout.flush()
    sys.stdout.write('\n')
    return nn

def binResults():
    # Bin Sigmoid Predictions - Threshold: 0.5
    predictions_bin = predictions.ravel()
    for x in range(len(predictions_bin)):
        if predictions.ravel()[x] > 0.5:
            predictions_bin[x] = int(1)
        elif predictions.ravel()[x] <= 0.5:
            predictions_bin[x] = int(0)
    return predictions_bin

def results(y_test,predictions):
    mismatches = 0
    # Count Mismatches
    for x,y in zip(y_test.ravel(),predictions):
        if int(x) != int(y):
            mismatches += 1 
    # Print Results
    print 'Mismatches: ', str(mismatches)
    print 'Accuracy: ', 1 - (float(mismatches)/len(y_test))
    print 'Test Size, ', str(len(y_test))
    return 1 - (float(mismatches)/len(y_test))
            
""" Main Driver to Call Above Functions """
if __name__ == '__main__':
    #Load Data
    loadDataI()
    test_results = []
    train_results = []
    target_names = ['Attack','Normal']
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=(1-0.8),train_size=(0.8),shuffle=True)
    
    """ Random Hill Search """
    print 'Running Random Hill Search:'
    nn = randomHillSearch(no_resets=100)
    print 'Test set Scores'
    predictions = nn.predict(X_test)
    predictions_binned = binResults()
    prediction_result= results(y_test,predictions_binned)
    print(classification_report(y_test, predictions_binned, target_names=target_names))
    print 'Train set Scores'
    predictions = nn.predict(X_train)
    predictions_binned = binResults()
    prediction_result= results(y_train,predictions_binned)
    print '--------------------------'
    

    """ Simulated Annealing """
    print 'Running Simulated Annealing:'
    nn = simulatedAnnealing(k=30, temperature=1.0)
    print 'Test set Scores'
    predictions = nn.predict(X_test)
    predictions_binned = binResults()
    prediction_result= results(y_test,predictions_binned)
    print(classification_report(y_test, predictions_binned, target_names=target_names))
    print 'Train set Scores'
    predictions = nn.predict(X_train)
    predictions_binned = binResults()
    prediction_result= results(y_train,predictions_binned)
    print '--------------------------'

    """ Genetic Algos """
    print 'Running Genetic Algorithm'
    print 'Please Wait...'
    from optimization import gaNN
    nn = NeuralNetwork()
    nn = gaNN(nn,X_train,y_train,no_generations=200,pop_size=10)
    print 'Test set Scores'
    predictions = nn.predict(X_test)
    predictions_binned = binResults()
    prediction_result = results(y_test,predictions_binned)
    test_results.append(prediction_result)
    print(classification_report(y_test, predictions_binned, target_names=target_names))
    print 'Train set Scores'
    predictions = nn.predict(X_train)
    predictions_binned = binResults()
    prediction_result= results(y_train,predictions_binned)
    train_results.append(prediction_result)
    print '--------------------------'


    
    
    
  
    





    
    








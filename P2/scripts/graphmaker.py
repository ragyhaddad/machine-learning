#!/bin/bash/python
import matplotlib.pyplot as plt
import numpy as np 

x_axis = []


""" This Script was used to plot the different Graphs in my analysis """
##Graph 1 RHS Resets
test = [0.8694999999999999, 0.9535, 0.8694999999999999, 0.8694999999999999, 0.971, 0.9665, 0.8694999999999999, 0.8694999999999999, 0.9705, 0.9390000000000001, 0.969, 0.9635, 0.9410000000000001, 0.9585, 0.9735, 0.965,0.97, 0.8694999999999999, 0.953]
train = [0.8815, 0.95475, 0.8815, 0.881625, 0.970875, 0.9645, 0.8815, 0.8815, 0.96825, 0.943, 0.965125, 0.964375, 0.947, 0.964375, 0.97325, 0.97, 0.96675, 0.8815, 0.956875]
x_axis = np.arange(10,200,10)

##RHS train size graph
test = [0.9692222222222222, 0.970125, 0.9707142857142858, 0.9675, 0.9722, 0.96775, 0.8893333333333333, 0.9595, 0.897]
train = [0.969, 0.9635, 0.9676666666666667, 0.97275, 0.9694, 0.9701666666666666,0.8921428571428571, 0.9695, 0.9084444444444444]
x_axis = np.arange(0.1,1,0.1)


#Graph 2 SA number of K 
test = [0.881, 0.9705, 0.9665, 0.9845, 0.9835, 0.985]
train = [0.878625, 0.965625, 0.965625, 0.979125, 0.979375, 0.984375]


## Graph 2 SA train Size 
test = [0.9824444444444445, 0.98375, 0.9838571428571429, 0.9825, 0.9818, 0.98425, 0.9866666666666667, 0.983, 0.98]
train = [0.989, 0.984, 0.9863333333333333, 0.98675, 0.9842, 0.9841666666666666, 0.9838571428571429, 0.9855, 0.9795555555555555]
x_axis = np.arange(0.1,1.0,0.1)

##Genetic algos generation size
test = [0.9675, 0.8825000000000001, 0.8825000000000001, 0.9625, 0.8825000000000001, 0.12150000000000005, 0.888,0.9655, 0.08850000000000002, 0.8825000000000001, 0.9365, 0.966, 0.9299999999999999, 0.11699999999999999,0.029000000000000026, 0.9295, 0.8825000000000001, 0.9635, 0.884, 0.9695]
train = [0.967625, 0.87825, 0.8785000000000001, 0.96475, 0.87825, 0.12324999999999997, 0.883375, 0.97, 0.08374999999999999, 0.87825, 0.93525, 0.9645, 0.926375, 0.12062499999999998, 0.03149999999999997, 0.91975, 0.87825, 0.9645, 0.87925, 0.971625] 
x_axis = np.arange(1,200,10)


test = [0.8795, 0.12050000000000005, 0.09899999999999998, 0.958, 0.8795, 0.121] 
train = [0.879, 0.121, 0.09824999999999995, 0.955375, 0.879, 0.12024999999999997] 
x_axis = np.arange(5,60,10)


plt.ylim(0,1)
def plot(title='none',x=x_axis,test_set=test,train_set = train):
    plt.grid(linestyle='dotted')
    plt.title(title)
    plt.xlabel('Population Size')
    plt.ylabel('Accuracy')
    plt.plot(x_axis,test,'-o',color='#235dba',alpha=1,label='Test Set')
    plt.plot(x_axis,train,'-o',color='#e56e19',label='Train Set',alpha=0.9)
    plt.legend(loc='best')
    plt.savefig('../graphs/ga-2-popsize.png',dpi=300)
    plt.show()

plot(title='Accuracy VS Population Size(GA)')
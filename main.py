import matplotlib.pyplot as plt;import pandas as pd;import random ;import os; import numpy as np
def sigmoid(z):    return 1/ (1 + np.exp(-z))
def Grad(theta , learningRate , feature , class1):
    theta = np.matrix(theta);feature = np.matrix(feature);    target = np.matrix(class1)
    deltaD = np.matrix([1.0,1.0,1.0,1.0,1.0])
    delta = np.matrix([0.00005,0.00005,0.00005,0.00005,0.00005])
    counter = 0
    while np.greater_equal(np.abs(deltaD) , delta ).any():
        counter += 1
        cost1 = sigmoid(feature * theta.T) - class1
        for i1 in range(0 , 5):
            temp = np.multiply( cost1 , feature[ : , i1])
            t4 =  float(learningRate  * np.sum( temp ) / len(class1))
            deltaD[0 , i1 ] = -t4
        theta = np.add(theta , deltaD)
    if counter!= 0 :return [theta , counter]
def CV(k , alpha , theta):
    counter = [] ; precision =[]; recall =[]; F1_score =[];accuracy = []
    steps = np.arange(0 , 1 , 1.0/k)
    for i in steps:
        FP = 0 ; TP = 0; FN = 0; TN = 0
        first_point = int(i*dataset.__len__())
        second_point = int((i+(1.0/k))* dataset.__len__())
        FeatureTest = dataset.iloc[first_point : second_point  , 0:5]
        ClassTest = dataset.iloc[first_point : second_point  , 5:6]
        FeatureTrain = dataset.iloc[: , 0:5]
        ClassTrain = dataset.iloc[: , 5:6]
        FeatureTrain.drop(FeatureTrain.index[first_point : second_point] , inplace = True)
        ClassTrain.drop(ClassTrain.index[first_point : second_point] , inplace = True)
        [theta , count]= Grad(theta , alpha , FeatureTrain , ClassTrain)
        FeatureTest = np.matrix(FeatureTest)
        ClassTest = np.matrix(ClassTest)
        predict = sigmoid(FeatureTest * theta.T)
        for i in range(0, FeatureTest.__len__()):
            if (float(predict.item(i)) >= 0.5) and (int(ClassTest.item(i)) == 1):                TP += 1
            elif (float(predict.item(i)) >= 0.5) and (int(ClassTest.item(i)) == 0):                FP += 1
            elif (float(predict.item(i)) < 0.5) and (int(ClassTest.item(i)) == 1):                FN += 1
            elif (float(predict.item(i)) < 0.5) and (int(ClassTest.item(i)) == 0):                TN += 1
        counter.append(count)
        precision.append(TP / (TP + FP))
        recall.append(TP / (TP + FN))
        F1_score.append((TP / (TP + FP)) * (TP / (TP + FN)) * 2 / ((TP / (TP + FP)) + (TP / (TP + FN))))
        accuracy.append((TN + TP) / (TN+TP+FN+FP))
    return [np.array(accuracy).mean() , np.array(precision).mean() , np.array(recall).mean() , np.array(F1_score).mean() ,np.array(counter).mean()]
def cost(theta , i, j):
    theta = np.matrix(theta)
    i = np.matrix(i) ;  j = np.matrix(j)
    first_part = np.multiply(-j, np.log10(sigmoid(i * theta.T)))
    second_part = np.multiply((1-j) , np.log10(1 - sigmoid(i * theta.T)))
    return np.sum(first_part - second_part) / len(i)

dataset = pd.read_csv('data_banknote_authentication.txt' , names = ['f1' , 'f2' , 'f3' , 'f4' , 'class'])
a = np.matrix(np.ones(len(dataset))).T
dataset = np.append(a , np.matrix(dataset) , axis = 1)
dataset = pd.DataFrame(dataset)
index = list(dataset.index)
random.shuffle(index)
dataset = dataset.ix[index]

Inter1 = [ 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1. ]
Inter1 = np.array(Inter1)
P1 = []
for a in Inter1:
    theta = [1.0,1.0,1.0,1.0,1.0]
    P1.append(CV(5, a, theta))
P1 = np.matrix(P1)

f, axarr = plt.subplots(2, 3)
axarr[0, 0].plot(np.array(Inter1), np.array(P1[:, 0]) , 'r')
axarr[0, 0].set_title('accuracy')
axarr[0, 1].plot(np.array(Inter1), np.array(P1[:, 1]) , 'b')
axarr[0, 1].set_title('precision')
axarr[0, 2].plot(np.array(Inter1), np.array(P1[:, 2]) , 'g')
axarr[0, 2].set_title('recall')
axarr[1, 0].plot(np.array(Inter1), np.array(P1[:, 3]) , 'y')
axarr[1, 0].set_title('F1 score')
axarr[1, 1].plot(np.array(Inter1), np.array(P1[:, 4]) , 'k')
axarr[1, 1].set_title('counter')
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()




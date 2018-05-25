import numpy as np
import pandas as pd
import os
import math

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
def heuristic(instansVec,tetVec):
    sum1 = 0.0
    for i,row  in enumerate(instansVec):
        sum1 = sum1 + float(row)* tetVec[i]
        #print(str(row)+ '       =======      '+str(tetVec[i]))
    return sum1
def predict(theta, X):
    print(theta)
    probability = sigmoid(X * theta.T)
    #print(sigmoid(X * theta.T))
    return [1 if X >= 0.5 else 0 for X in probability]
def findTheta(X1,Y1,alfa,tet):
    threshould = 0.01
    """
    #Dataset = list(zip(X,Y))
    #np.random.shuffle(Dataset)
    #cols = data.shape[1]
    #X1 = data.iloc[:, 0:cols - 1]
    #Y1 = data.iloc[:, cols - 1:cols]
    """
    bool =True
    c=0
    while(bool):
        h1 = []
        new_tet = [0.0,0.0,0.0,0.0,0.0]
        for i in range(0,len(Y1)):
            h1.append(sigmoid(heuristic(X1[i],tet)))
        if (len(h1)!= len(Y1) ):print('Error in Size')
        for i in range(0, len(tet)):
            sum2 = 0
            for j in range(0,len(h1)):
                #print(str(i) +  '      ' + str(j) + '      ' + str(len(tet))+ '        ' + str(len(Y1)) + '       h1  ' + str(len(h1))+ '   h1[0]  '+str(h1[0])+'   '+str(X1[j][i])+ '   h1[j]= '+str(h1[j])+'      y[j]='+str( float(Y1[j])) )
                #print(str(i) +  '      ' + str(j) + '    ' + str(ds[j][i]))
                #print( str(j)+'    ' +str(float(h1[j] - y[j])* float(ds[j][i])) + '         ' + str(sum2))
                sum2 += float(h1[j] - Y1[j])* float(X1[j][i])
            new_tet[i] = float(tet[i] - (alfa * sum2))


        #calc distance of new teta and old teta
        bool = False
        for i in range(0, len(tet)):
            if math.fabs(tet[i] - new_tet[i]) > threshould: bool = True
        tet=new_tet
        c+=1
    print('alfa = '+str(alfa)+' => number of iter :'+str(c))
    return tet ,c
def CV5fold(x,y,alfa):
    result =[]
    result.append(
        ['round', 'Train size', 'Test size', 'Real Positive', 'Real Negative', 'TP', 'TN', 'FP', 'FN', 'Accuracy','TPR (Recall)', 'Percision', 'F1 Score','alfa','iteration'])
    for i in range(1,6):
        #update train and test ds
        test_X = [] ; train_X = [] ;test_Y = [] ; train_Y = []

        for j in range ( 0 , len(x)):
            if j < round((i/5)*len(x)) and j >= (round(((i-1)/5)*len(x))):
                test_X.append(x[j])
                #print(str(i)+ '    ' +str(j)+ '    test')
            else:
                train_X.append(x[j])
                #print(str(i) + '    ' + str(j) + '    train')

        for j in range(0 , len(y)):
            if j < round((i/5)*len(y)) and j >= (round(((i-1)/5)*len(y))):
                test_Y.append(y[j])
            else:
                train_Y.append(y[j])

        theta,c = findTheta(train_X, train_Y, alfa, np.ones(5))
        theta_min = np.matrix(theta)
        predictions = predict(theta_min,test_X)
        #result = ['round','Real Positive','Real Negative','TP','TN','FP','FN','Accuracy' , 'TPR (Recall)','Percision','F1 Score']
        TP = [1 if (a == 1 and b == 1)  else 0 for (a, b) in zip(predictions, test_Y)]
        TN = [1 if (a == 0 and b == 0)  else 0 for (a, b) in zip(predictions, test_Y)]
        FP = [1 if (a == 1 and b == 0)  else 0 for (a, b) in zip(predictions, test_Y)]
        FN = [1 if (a == 0 and b == 1)  else 0 for (a, b) in zip(predictions, test_Y)]
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, test_Y)]
        accuracy = (sum(map(int, correct)) % len(correct))
        result.append([i,      #round
                      len(train_Y),  #Train size
                      len(test_Y),    #Test size
                      sum(map(int,test_Y)) ,        # Real Positive
                      len(test_Y)- sum(map(int,test_Y)),    #Real Negative
                      sum(map(int,TP)),   #TP
                      sum(map(int,TN)),   #TN
                      sum(map(int,FP)),   #FP
                      sum(map(int,FN)),   #FN
                      accuracy * 100 / len(test_Y),  #Accuracy
                      sum(map(int,TP))/  sum(map(int,test_Y)),    #Recall
                      sum(map(int,TP))/(sum(map(int,TP))+sum(map(int,FP))) ,    #Percision
                      (2*sum(map(int,TP)))/((2*sum(map(int,TP)))+ sum(map(int,FP))+  sum(map(int,FN)) ), #F1 score
                        alfa,   #alfa (learning rate)
                      c])       # number of iteration

        #print('accuracy = {0}%'.format(accuracy * 100 / len(test_Y)))
    return result
def feedforwardCV5fold(x,y,alfa):
    result =[]
    result.append(['round','Train size','Test size','Real Positive','Real Negative','TP','TN','FP','FN','Accuracy' , 'TPR (Recall)','Percision','F1 Score','Alfa (learning rate)','itteration'])
    for i in range(1,5):
        #update train and test ds
        test_X = [] ; train_X = [] ;test_Y = [] ; train_Y = []

        for j in range ( 0 , len(x)-1):
            if j < round((i/5)*len(x)) and j >= 0 :
                train_Y.append(y[j])
                train_X.append(x[j])
            elif  j<round(((i+1)/5)*len(x)):
                test_X.append(x[j])
                test_Y.append(y[j])


        theta,c = findTheta(train_X, train_Y, alfa, np.ones(5))
        theta_min = np.matrix(theta)
        predictions = predict(theta_min,test_X)
        #result = ['round','Real Positive','Real Negative','TP','TN','FP','FN','Accuracy' , 'TPR (Recall)','Percision','F1 Score']
        TP = [1 if (a == 1 and b == 1)  else 0 for (a, b) in zip(predictions, test_Y)]
        TN = [1 if (a == 0 and b == 0)  else 0 for (a, b) in zip(predictions, test_Y)]
        FP = [1 if (a == 1 and b == 0)  else 0 for (a, b) in zip(predictions, test_Y)]
        FN = [1 if (a == 0 and b == 1)  else 0 for (a, b) in zip(predictions, test_Y)]
        correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, test_Y)]
        accuracy = (sum(map(int, correct)) % len(correct))
        result.append([i,      #round
                      len(train_Y),  #Train size
                      len(test_Y),    #Test size
                      sum(map(int,test_Y)) ,        # Real Positive
                      len(test_Y)- sum(map(int,test_Y)),    #Real Negative
                      sum(map(int,TP)),   #TP
                      sum(map(int,TN)),   #TN
                      sum(map(int,FP)),   #FP
                      sum(map(int,FN)),   #FN
                      accuracy * 100 / len(test_Y),  #Accuracy
                      sum(map(int,TP))/  sum(map(int,test_Y)),    #Recall
                      sum(map(int,TP))/(sum(map(int,TP))+sum(map(int,FP))) ,    #Percision
                      (2*sum(map(int,TP)))/((2*sum(map(int,TP)))+ sum(map(int,FP))+  sum(map(int,FN)) ), #F1 score
                        alfa,    #alfa
                      c])       # number of itteration
        #print('accuracy = {0}%'.format(accuracy * 100 / len(test_Y)))
    return result
address = '\\data_banknote_authentication.txt'
path = os.getcwd() + address
data = pd.read_csv(path, header=None, names=['F1','F2', 'F3','F4', 'C'])
data.head()

data.insert(0, 'Ones', 1)    # add a ones column - this makes the matrix multiplication work out easier    => Ezafeh kardan soton 1 ha


data = data.reindex(np.random.permutation(data.index))

cols = data.shape[1]
X = data.iloc[:,0:cols-1]  # set X (training data)
y = data.iloc[:,cols-1:cols]  # set  y (target variable)

alfa =0.000001



X = np.array(X.values)  # convert to numpy arrays
y = np.array(y.values)  # convert to numpy arrays


theta,c = findTheta(X,y,alfa,np.ones(5))    #  initalize the parameter array theta
cost(theta, X, y)
theta_min = np.matrix(theta)
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy*100/len(y)))


f1 =open(address.replace('.txt','result .csv').replace('\\',''),'w')

    #print(line)

f1.write('Feed Forward Cros Validation\n')
ls = feedforwardCV5fold(X,y,alfa)

for line in ls :
    f1.write(str(line).replace("'",'').replace('[','').replace(']','')+'\n')
print('=================================')
f1.write('Cros Validation\n')
ls = CV5fold(X,y,alfa)
for line in ls :
    f1.write(str(line).replace("'",'').replace('[','').replace(']','')+'\n')


f2 = open('Learning rate1.csv','r')
dd = []
for line in f2:
    dd.append(line)
f3 = open('Learning rate1.csv','w')
bool = True
for l in dd :
    f3.write(l)
    str1=str(l).split(',')
    if str1[0].strip()==str(alfa):bool=False
print(str(ls[1][9])+'    '+str(ls[2][9])+ '      '+str(ls[3][9])+'        '+str(ls[4][9])+ '        '+str(ls[5][9]))
acc = (ls[5][9]+ls[1][9]+ls[2][9]+ls[3][9]+ls[4][9])/5
rec = (ls[5][10]+ls[1][10]+ls[2][10]+ls[3][10]+ls[4][10])/5
perc = (ls[5][11]+ls[1][11]+ls[2][11]+ls[3][11]+ls[4][11])/5
count = (ls[5][14]+ls[1][14]+ls[2][14]+ls[3][14]+ls[4][14])/5
if(bool):
    f3.write(str([alfa,count,acc,rec,perc]).replace(']','').replace('[','')+'\n')

f1.close() ; f2.close();f3.close()
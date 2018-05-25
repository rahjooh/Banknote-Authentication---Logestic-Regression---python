import numpy as np
import math

def sigmoid(zVar):
    return 1 / (1 + np.exp(-zVar))

def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def heuristic(instansVec,tetVec):
    sum1 = 0.0
    for i,row  in enumerate(instansVec):
        sum1 = sum1 + float(row)* tetVec[i]
        #print(str(row)+ '       =======      '+str(tetVec[i]))
    return sum1

def logesticTrain(ds,alfa,tet):
    threshould = 0.001
    counter = 0
    np.random.shuffle(ds)
    while(True):
        print('1',end='')
        h1 = []
        y = []
        new_tet = [0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(0,len(ds)):
            h1.append(sigmoid(heuristic(ds[i][0:5] , tet)))
            y.append(float(ds[i][5]))
        """print('++++++++++++++++++++++++++++++++++++++++++++++')
        print(tet)
        print(new_tet)
        print(h1)
        print(y)"""
        for i in range(0,len(tet)):
            sum2= 0

            for j in range(0,len(h1)):
                #print(str(i) +  '      ' + str(j) + '      ' + str(len(tet))+ '        ' + str(len(ds)) + '         ' + str(len(h1))+ '')
                #print(str(i) +  '      ' + str(j) + '    ' + str(ds[j][i]))
                #print( str(j)+'    ' +str(float(h1[j] - y[j])* float(ds[j][i])) + '         ' + str(sum2))
                sum2 += float(h1[j] - y[j])* float(ds[j][i])
            new_tet[i] = tet[i] - (alfa * sum2)

        #calc distance of new teta and old teta
        bool = True
        for i in range(0, len(tet)):
            if math.fabs(tet[i] - new_tet[i]) > threshould: bool = False

        if bool:
            break
        else:
            tet = new_tet

        counter+=1
        #print('              '+str(counter)+ ':' + str(new_tet))
    return tet

def logesticTest(ds,tet):
    acc = {}
    acc['TP'] = 0 ; acc['FP'] = 0 ; acc['TN'] = 0 ; acc['FN'] =0
    for i in range(0,len(ds)):
        #print(ds[i][0:5])
        h1 = sigmoid(heuristic(ds[i][0:5] , tet))
        #print(str(h1)+ '    ' + str(ds[i][5]),)
        if h1 > 0.5 :
            if ds[i][5] == '1':
                acc['TP'] +=1
               # print ('TP')
            else :
                acc['FP'] +=1
                #print('FP')
        else:
            if ds[i][5] == '1':
                acc['FN'] +=1
                #print('FN')
            else:
                acc['TN']+=1
               # print('TN')
    print(len(ds))
    for i in acc:
        print(str(i)+ '    '+ str(acc[i]))

def CV5fold(ds,tet,alfa):
    p  = [] ; n = []
    for l in ds :
        if l[5]=='1':
            p.append(l)
        else:
            n.append(l)

    for i in range(1,6):
        #update train and test ds
        test = [] ; train = []
        for j in range ( 0 , len(p)):
            if j < round((i/5)*len(p)) and j >= (round(((i-1)/5)*len(p))):
                test.append(p[j])
                #print(str(i)+ '    ' +str(j)+ '    test')
            else:
                train.append(p[j])
                #print(str(i) + '    ' + str(j) + '    train')

        for j in range(0 , len(n)):
            if j < round((i/5)*len(n)) and j >= (round(((i-1)/5)*len(n))):
                test.append(n[j])
            else:
                train.append(n[j])
        mytet = logesticTrain(train,alfa1,tet)
        logesticTest(train,mytet)
        print(mytet)


alfa1 = 0.05 ; ds1=[] ;tet1 = [ 1.0 , 1.0 , 1.0 , 1.0 , 1.0];  file1 = open('data_banknote_authentication.txt','r')
for line in file1:
    ds1.append(('1.0,' + line.replace('\n', '')).split(','))
np.random.shuffle(ds1)

CV5fold(ds1,tet1,alfa1)
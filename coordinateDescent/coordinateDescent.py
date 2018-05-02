'''
Created on 18-Oct-2017

@author: Ashwin
'''
import csv
from math import exp

import numpy as np


def readTrainData():
    File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data")
    reader = csv.reader(File)
    X = []
    Y = []
    for row in reader:
        X.append([x for x in row[1:]])
        for x in row[0]:
            if x == '0':
                Y.append("-1")
            else:
                Y.append("1")        
    File.close()
    return (X,Y)

def readTestData():
    File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_test.data")
    reader = csv.reader(File)
    X = []
    Y = []
    for row in reader:
        X.append([x for x in row[1:]])
        for x in row[0]:
            if x == 0:
                Y.append("-1")
            else:
                Y.append("1") 
    File.close()
    return (X,Y)


class Node:
    def __init__(self,i):
        self.val = i
        self.leftChild = None
        self.rightChild = None    


def formHypothesis(X, Y):
    hyp = []
    m = len(X[0])
    for i in range(m):
        for leftChild in ["0","1"]:
            for rightChild in ["0","1"]:
                node = Node(i)
                node.leftChild = str(leftChild)
                node.rightChild = str(rightChild)
                hyp.append(node)
    return hyp

def predict(h,X):
        temp = h
        prediction = None
        while True:
            if X[temp.val] == "0":
                temp = temp.leftChild
            else:
                temp = temp.rightChild
             
            if  temp == "1":
                prediction = temp
                break
            elif temp == "0":
                prediction = "-1"
                break    
        return prediction

def expLossFnt(hyp, h, alpha, x, y):
    suml = 0
    for i in range(len(hyp)):
        if hyp[i].val != h.val:
            suml += alpha[i] * int(predict(hyp[i],x))
    
    t = -1 * float(y) * float(suml)        
    return exp(t)

def classifier(hyp, alpha, x, y):
    suml = 0
    for i in range(len(hyp)):
            suml += alpha[i] * int(predict(hyp[i],x))
    
    t = -1 * float(y) * float(suml)        
    return t/abs(t)

        
def coordinateDescent(hyp, X,Y):
    alpha = [0] * len(hyp)
    final_hyp = [False] * len(hyp)
    k = 0
    while True:
        for h in hyp:
            print(alpha)
            nr = 0
            dr = 0
            for i in range(len(X)):
                if Y[i] == predict(h, X[i]):
                    nr += expLossFnt(hyp, h, alpha, X[i], Y[i])       
                else:
                    dr +=expLossFnt(hyp, h, alpha, X[i], Y[i])         
            alp = (0.5) *  np.log(nr/dr)
            if alp == alpha[h.val]:
                final_hyp[h.val]=True
            alpha[h.val] = alp           
        k = k+1
        if set(final_hyp) == True:
            break
        if k==200:
            break
    print(alpha)
    return alpha
            
def runCoordinateDescent():
    [X,Y] = readTrainData()
    hypothesis = formHypothesis(X,Y)
    #alpha = coordinateDescent(hypothesis, X, Y)
    #alpha = [-5.3290705182007798e-15, -1.6386318865366014, -8.9928064994638484e-15, -0.59146173792212531, -2.1926904736347321e-14, -0.50792526806868543, 3.8857805861880329e-15, -0.4277304268049541, -2.2037927038809843e-14, -3.4286907161317135, -9.9920072216265083e-15, -2.1570259940112289, -3.3695268797374637e-14, 0.77703725615277885, 2.0872192862952508e-14, 0.50102391255603274, -8.6597395920762967e-15, 0.31136222889956716, -4.218847493575613e-15, 0.13853548494773218, -9.5479180117764378e-15, 0.15450414166138821, -5.5511151231257857e-16, 0.1234990253973231, 1.665334536937732e-15, 0.61001304415942814, 1.2323475573339086e-14, 0.32400198105487921, -6.6058269965197248e-15, 3.0710999518581117, 1.8318679906314748e-14, 2.004465153783495, -6.5503158452884662e-15, -1.8525066635164977, -6.1062266354383649e-16, -1.3813713232460214, -2.8643754035329859e-14, 1.5115543168435517, 2.0983215165415017e-14, 0.58092361604777742, -1.2934098236883241e-14, 0.3791364922882422, 2.8865798640253987e-15, 0.24965196661839684, -2.1094237467878018e-15, -0.44781167687110174, -4.4408920985006459e-15, -0.27352498978326079, -6.1617377866696567e-15, 0.79687562612895146, 1.0547118733938875e-14, 0.38759568688335222, -1.8318679906315118e-15, 1.3346618135808024, 1.2767564783189138e-14, 1.0046650220381743, -6.6058269965197248e-15, 0.37948568369491964, -2.9976021664879317e-15, 0.36813725485640753, -3.3306690738754706e-16, 0.72471260159804984, 1.2323475573339086e-14, 0.58160665803789158, -9.1038288019263673e-15, 2.2558682572738262, 9.8809849191637954e-15, 2.1949185625166368, -2.720046410331641e-15, 1.4818929241110095, 1.5543122344752168e-15, 1.4633125140851524, -1.3267165144270797e-14, 0.054336604917530854, -2.1094237467878018e-15, 0.035577268206860878, -4.4408920985006459e-15, 0.30023825167010237, 4.6629367034256354e-15, 0.19711913757654898, -1.4988010832439635e-15, 0.1744646216885157, -1.0158540675320285e-14, 0.12059801139600147, -5.5511151231257857e-16, 0.35334145767447928, 4.9960036108131792e-15, 0.16248273644579239]
    alpha = [19.747676443600895, 48.136058706684295, 55.183195515146032, 44.712587750726641, 30.950179355319815, 39.741287437245091, 55.793087138865047, 19.808673062037318, 4.5019276880365204, 8.6573632962514111, 18.924944007912291, 16.520618654941973, -83.944303805253867, -153.20862481374166, -146.16148764140564, -51.68510246504998, -171.41315952058949, 288.34363343185191, 113.93790801105798, 66.919820717033872, 15.748733035808048, 22.073693158993219, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print(alpha)
    loss = 0
    for i in range(len(X)):
        loss += expLossFnt(hypothesis, Node(-1), alpha, X[i], Y[i])
    print(loss)
    c = 0
    for i in range(len(X)):
        print(Y[i])
        print(classifier(hypothesis, alpha, X[i], Y[i]))
        if int(Y[i]) == classifier(hypothesis, alpha, X[i], Y[i]):
            c = c+1  
    print(c)
    print(len(X))        
    print(c/len(X))   
runCoordinateDescent()
        
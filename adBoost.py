'''
Created on 15-Oct-2017

@author: Ashwin
'''

import csv
from math import log,exp,sqrt
import matplotlib.pyplot as plt

def trainingdata():
    File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data")
    reader = csv.reader(File)
    trainingData = []
    for row in reader:
        trainingData.append([x for x in row])
    File.close()
    return trainingData


def testingdata():
    File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_test.data")
    reader = csv.reader(File)
    testData = []
    for row in reader:
        testData.append([x for x in row])
    File.close()
    return testData


class node:
    def __init__(self,value):
        self.value = value
        self.leftChild = None
        self.rightChild = None

    def error(self,X):
        err = 0.0
        indexes = []
        for i in range(len(X)):
            point = X[i]
            prediction = self.predict(point)
            if prediction != point[0]:
                err = err + point[len(point) - 1]
                indexes.append(i)
        return (err,indexes)

    def predict(self,x):
        tree = self
        prediction = None
        while True:
            if x[tree.value] == "0":
                tree = tree.leftChild
            else:
                tree = tree.rightChild
            if tree == "0" or tree == "1":
                prediction = tree
                break
        return prediction


def oneDepthTree(noOfAttributes):
    trees = []
    for i in range(noOfAttributes):
        for c1 in ["0","1"]:
            for c2 in ["0", "1"]:
                n = node(i+1)
                n.leftChild = c1
                n.rightChild = c2
                trees.append(n)
    return trees

def twoDepthTree(attributelen,oneDepthTree):
    trees = []
    for i in range(attributelen):
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                n = node(i+1)
                n.leftChild = c1
                n.rightChild = one
                trees.append(n)
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                n = node(i+1)
                n.leftChild = one
                n.rightChild = c1
                trees.append(n)
    return trees

def threeDepthTree(attributelen,oneDepthTree,twoDepthTree):
    trees = []
    for i in range(attributelen):
        for one in oneDepthTree:
            for two in oneDepthTree:
                n = node(i+1)
                n.leftChild = one
                n.rightChild = two
                trees.append(n)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                n = node(i+1)
                n.leftChild = c1
                n.rightChild = one
                trees.append(n)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                n = node(i+1)
                n.leftChild = one
                n.rightChild = c1
                trees.append(n)
    return trees

def formHyp(train_datapoints):
    attributelen = len(train_datapoints[1]) - 1
    depth1Tree = oneDepthTree(attributelen)
    depth2Tree = twoDepthTree(attributelen,depth1Tree)
    depth3Tree = threeDepthTree(attributelen,depth1Tree,depth2Tree)
    return depth3Tree


def printResults(ht, at, trainData, testData):
    train = []
    test = []
    for i in range(len(ht)):
        h = ht[:(i+1)]
        a = at[:(i+1)]
        print(a)
        print("Number of rounds of boosting = "+str(i+1))
        acc_train = accuracy(h,a,trainData)
        train.append(acc_train)
        print("for training data accuracy = "+str(acc_train))
        acc_test = accuracy(h,a,testData)
        test.append(acc_test)
        print("for testing data accuracy = "+str(acc_test))
    return (train,test)    

def adaboost(M,train_datapoints,threeDepthTree):
    ht = []
    at = []
    for i in range(M):
        besttree = None
        besterr = float(1)
        errindexes = []
        for tree in threeDepthTree:
            err,indexes = tree.error(train_datapoints)
            if err < besterr:
                besterr = err
                besttree = tree
                errindexes = indexes
        a = float(0.5) * log(float(1-besterr)/float(besterr))
        ht.append(besttree)
        at.append(a)
        for j in range(len(train_datapoints)):
            point = train_datapoints[j]
            w_old = point[len(point) - 1]
            w_new = 1
            if errindexes.count(j) >0:
                w_new = (w_old * exp(a))/(2 * sqrt((1 - besterr) * besterr))
            else:
                na = -1 * a
                w_new = (w_old * exp(na)) / (2 * sqrt((1 - besterr) * besterr))
            train_datapoints[j][len(point) - 1] = w_new
    return (ht,at)


def accuracy(ht,at,datapoints):
    correct = 0
    for point in datapoints:
        sum1 = 0
        for i in range(len(ht)):
            pre = 0
            prediction = ht[i].predict(point)
            if prediction == "0":
                pre = -1
            else:
                pre = 1
            sum1 = sum1 + (at[i] * pre)
        sign = sum1/abs(sum1)
        if sign == -1:
            prediction = "0"
        else:
            prediction = "1"
        if prediction == point[0]:
            correct = correct + 1
    acc = float(correct)/float(len(datapoints))
    return acc


        
def runAdaboost():
    trainData = trainingdata()
    testData =  testingdata()
    weights = float(1)/float(len(trainData))
    
    updatedTrainingData = trainData
    for x in range(len(updatedTrainingData)):
        updatedTrainingData[x].append(weights)
    
    ht,at = adaboost(10,updatedTrainingData,formHyp(trainData))
    #ht,at = adaboost(20,updatedTrainingData,oneDepthTree(len(trainData[1]) - 1))
    print('done')
    
    [train, test] = printResults(ht,at,trainData,testData)
    
    
    #plot
    plt.plot( train, 'b', label="trainingDataSet")
    plt.plot( test, 'r', label="testDataSet")
    plt.legend()
    plt.show()
    
    
    
runAdaboost()
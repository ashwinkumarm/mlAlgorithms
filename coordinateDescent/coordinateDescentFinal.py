'''
Created on 20-Oct-2017

@author: Ashwin
'''

import csv
from math import log, exp

import numpy as np


def trainingdata():
    File = open("heart_train.data")
    reader = csv.reader(File)
    train_datapoints = []
    for row in reader:
        train_datapoints.append([x for x in row])
    File.close()
    return train_datapoints


def testingdata():
    File = open("heart_test.data")
    reader = csv.reader(File)
    test_datapoints = []
    for row in reader:
        test_datapoints.append([x for x in row])
    File.close()
    return test_datapoints


class node:
    def _init_(self,ind):
        self.value = ind
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
        temp = self
        prediction = None
        while True:
            if x[temp.value] == "0":
                temp = temp.leftChild
            else:
                temp = temp.rightChild
            if temp == "0" or temp == "1":
                prediction = temp
                break
        return prediction


def oneDepthTree(noOfAttributes):
    trees = []
    for i in range(noOfAttributes):
        for c1 in ["0","1"]:
            for c2 in ["0", "1"]:
                temp = node(i+1)
                temp.leftChild = c1
                temp.rightChild = c2
                trees.append(temp)
    return trees

def twoDepthTree(attributelen,oneDepthTree):
    trees = []
    for i in range(attributelen):
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                temp = node(i+1)
                temp.leftChild = c1
                temp.rightChild = one
                trees.append(temp)
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                temp = node(i+1)
                temp.leftChild = one
                temp.rightChild = c1
                trees.append(temp)
    return trees

def threeDepthTree(attributelen,oneDepthTree,twoDepthTree):
    trees = []
    for i in range(attributelen):
        for one in oneDepthTree:
            for two in oneDepthTree:
                temp = node(i+1)
                temp.leftChild = one
                temp.rightChild = two
                trees.append(temp)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                temp = node(i+1)
                temp.leftChild = c1
                temp.rightChild = one
                trees.append(temp)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                temp = node(i+1)
                temp.leftChild = one
                temp.rightChild = c1
                trees.append(temp)
    return trees


def recursive_print(root = node(1),p = "",dire = ""):
    print "parent "+p+" direction "+ dire+" cnode "+str(root.value)
    left = root.leftChild
    if type(left) == type("asd"):
        print "parent "+str(root.value)+" direction leftChild "+" cnode "+left
    else:
        recursive_print(left,str(root.value),"leftChild")
    right = root.rightChild
    if type(right) == type("asd"):
        print "parent "+str(root.value)+" direction rightChild "+" cnode "+right
    else:
        recursive_print(right,str(root.value),"rightChild")




train_datapints = trainingdata()
test_datapoints = testingdata()

def eloss(p,matrix,alpha):
    sum = float(0)
    for i in range(len(alpha)):
        sum = sum + alpha[i] * float(matrix[i][p])
    return exp(-1 * sum)

def elossTotal(datapoints,matrix,alpha):
    sum = float(0)
    for i in range(len(datapoints)):
        sum = sum + eloss(i,matrix,alpha)
    return sum


def formatedInp(trees,data):
    final_matrix = []
    for tree in trees:
        tree_matrix = []
        for point in data:
            # class of point
            clss = point[0]
            cl = -1
            if clss == "0":
                cl = -1
            else:
                cl = 1
            # predicted class of point
            prediction = tree.predict(point)
            pre = -1
            if prediction == "0":
                pre = -1
            else:
                pre = 1
            tree_matrix.append(cl * pre)
        final_matrix.append(tree_matrix)
    return final_matrix

def guess(c,p,inp,alpha):
    sum = float(0)
    for i in range(len(alpha)):
        if i != c:
            sum = sum + alpha[i] * float(inp[i][p])

    return exp(-1 * sum)


def classifier(c, formatedInp ,indexes,alpha):
    sum = float(0)
    for p in indexes:
        p_sum = guess(c,p,formatedInp,alpha)
        sum = sum + p_sum
    return sum

def coordinateDescent(trees,datapoints,matrix):
    alpha = [float(0) for x in trees]
    add = []
    for xyz in range(200):
        for i in range(len(alpha)):
            predictions = matrix[i]
            np_predictions = np.array(predictions)
            correct = list(np.where( np_predictions == 1)[0])
            wrong = list(np.where( np_predictions == -1)[0])
            numerator = classifier(i,matrix,correct,alpha)
            denominator = classifier(i,matrix,wrong,alpha)
            alpha[i] = 0.5 * log(float(numerator)/float(denominator))
        if (xyz+1) % 5 == 0:
            acc = accuracy(matrix,alpha,datapoints)
            add.append(acc)
    print elossTotal(datapoints,matrix,alpha)
    add.sort()
    print add
    return alpha


def predict(matrix,alpha,i):
    sum = float(0)
    for a in range(len(alpha)):
        sum = sum + alpha[a] * matrix[a][i]
    ret = sum/abs(sum)
    if ret == -1:
        return "0"
    else:
        return "1"

def accuracy(matrix,alpha,datapoints):
    correct = 0
    for i in range(len(datapoints)):
        pred = predict(matrix,alpha,i)
        if pred == datapoints[i][0]:
            correct = correct +1
    return float(correct)/float(len(datapoints))


oneatttrees = oneDepthTree(len(train_datapints[0]) - 1)
matrix = formatedInp(oneatttrees,train_datapints)

alpha = coordinateDescent(oneatttrees,train_datapints,matrix)
print accuracy(matrix,alpha,train_datapints)



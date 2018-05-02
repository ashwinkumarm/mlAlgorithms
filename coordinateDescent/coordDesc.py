'''
Created on 20-Oct-2017

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
        Y.append([x for x in row[0]])
    File.close()
    return (X, Y)


def readTestData():
    File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_test.data")
    reader = csv.reader(File)
    X = []
    Y = []
    for row in reader:
        X.append([x for x in row[1:]])
        Y.append([x for x in row[0]])
    File.close()
    return (X, Y)


class Node:
    def __init__(self, i):
        self.val = i
        self.leftChild = None
        self.rightChild = None


def formHypothesis(X, Y):
    hyp = []
    m = len(X[0])
    for i in range(m):
        for leftChild in ["0", "1"]:
            for rightChild in ["0", "1"]:
                node = Node(i)
                node.leftChild = str(leftChild)
                node.rightChild = str(rightChild)
                hyp.append(node)
    return hyp


def predict(h, X):
    temp = h
    prediction = None
    while True:
        if X[temp.val] == "0":
            temp = temp.leftChild
        else:
            temp = temp.rightChild
        if temp == "0" or temp == "1":
            prediction = temp
            break
    return prediction


def expLossFnt(hyp, h, alpha, x, y):
    suml = 0
    for i in range(len(hyp)):
        if hyp[i].val != h.val:
            if int(predict(hyp[i], x)) == 0:
                p = -1
            else:
                p = 1
            suml += alpha[i] * p
    c = -1
    if y[0] == 0:
        c = -1
    else:
        c = 1
    t = -1 * float(c) * float(suml)
    return exp(t)


def coordinateDescent(hyp, X, Y):
    alpha = [0] * len(hyp)
    final_hyp = [False] * len(hyp)
    k = 0
    while True:
        for h in hyp:
            nr = 0
            dr = 0
            for i in range(len(X)):
                if Y[i][0] == predict(h, X[i]):
                    nr += expLossFnt(hyp, h, alpha, X[i], Y[i][0])
                else:
                    dr += expLossFnt(hyp, h, alpha, X[i], Y[i][0])
            alp = (0.5) * np.log(nr / dr)
            if alp == alpha[h.val]:
                final_hyp[h.val] = True
            alpha[h.val] = alp
        k = k + 1
        if set(final_hyp) == True:
            break
        if k == 200:
            break
    print(alpha)
    return alpha


def runCoordinateDescent():
    [X, Y] = readTrainData()
    hypothesis = formHypothesis(X, Y)
    alpha = coordinateDescent(hypothesis, X, Y)
    loss = 0
    for i in range(len(X)):
        loss += expLossFnt(hypothesis, Node(-1), alpha, X[i], Y[i])
    print(loss)
    print(accuracy(hypothesis, Node(-1), alpha, X, Y))

def classify(hyp, h, alpha, x):
    suml = 0
    for i in range(len(hyp)):
        if int(predict(hyp[i], x)) == 0:
            p = -1
        else:
            p = 1
        suml += alpha[i] * p
    sign = suml/abs(suml)
    if sign == -1:
        return "0"
    else:
        return "1"


def accuracy(hyp, h, alpha, X, Y):
    correct = 0
    for x in range(len(X)):
        pre = classify(hyp, h, alpha,X[x])
        if pre == Y[x][0]:
            correct = correct + 1
    return float(correct)/float(len(X))


alpha = runCoordinateDescent()

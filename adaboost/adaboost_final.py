'''
Created on 18-Oct-2017

@author: Ashwin
'''
import csv
from math import log, exp, sqrt


File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data")
reader = csv.reader(File)
train_datapoints = []
for row in reader:
    train_datapoints.append([x for x in row])
File.close()

File = open("D:\semester_3\machine learning\Assgn\Assgn3\heart_test.data")
reader = csv.reader(File)
test_datapoints = []
for row in reader:
    test_datapoints.append([x for x in row])
File.close()


class node:
    def __init__(self,value):
        self.value = value
        self.leftChild = None
        self.rightChild = None

    def error(self,X):
        err = 0
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
    for i in range(noOfAttributes -1):
        for c1 in ["0","1"]:
            for c2 in ["0", "1"]:
                temp = node(i+1)
                temp.leftChild = c1
                temp.rightChild = c2
                trees.append(temp)
    return trees


def twoDepthTree(attributelen,oneDepthTree):
    trees = []
    for i in range(attributelen -1):
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
    for i in range(attributelen -1):
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
        #print("round "+str(i + 1))
        #print(besterr)
        #recursive_print(besttree)
        # print log((1-besterr)/besterr)
        # print log(float(1-besterr)/float(besterr))
        # print float(0.5) * log(float(1-besterr)/float(besterr))
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
        # print (float(1) - besterr)
    return (ht,at)


def recursive_print(root = node(1),p = "",dire = ""):
    print("parent "+p+" direction "+ dire+" cnode "+str(root.value))
    left = root.leftChild
    if type(left) == type("asd"):
        print("parent "+str(root.value)+" direction leftChild "+" cnode "+left)
    else:
        recursive_print(left,str(root.value),"leftChild")
    right = root.rightChild
    if type(right) == type("asd"):
        print("parent "+str(root.value)+" direction rightChild "+" cnode "+right)
    else:
        recursive_print(right,str(root.value),"rightChild")
        
# def adaboost(M,train_datapoints,threeatttrees):
#     ht = []
#     at = []
#     for i in range(M):
#         besttree = None
#         besterr = 1
#         errindexes = []
#         for tree in threeatttrees:
#             err,indexes = tree.error(train_datapoints)
#             if err < besterr:
#                 besterr = err
#                 besttree = tree
#                 errindexes = indexes
#         a = float(1/2) * log((1-besterr)/besterr)
#         ht.append(besttree)
#         at.append(a)
#         for j in range(len(train_datapoints)):
#             point = train_datapoints[j]
#             w_old = point[len(point) - 1]
#             w_new = 1
#             if j in errindexes:
#                 w_new = (w_old * exp(a))/(2 * sqrt((1 - besterr) * besterr))
#             else:
#                 w_new = (w_old * exp(-1 * a)) / (2 * sqrt((1 - besterr) * besterr))
#             train_datapoints[j][len(point) - 1] = w_new
#         # print (float(1) - besterr)
#     return (ht,at)


def accuracy(ht,at,datapoints):
    correct = 0
    for point in datapoints:
        sum = 0
        for i in range(len(ht)):
            pre = 0
            prediction = ht[i].predict(point)
            if prediction == "0":
                pre = -1
            else:
                pre = 1
            sum = sum + at[i] * pre
        sign = sum/abs(sum)
        if sign == -1:
            prediction = "0"
        else:
            prediction = "1"
        if prediction == point[0]:
            correct = correct + 1
    acc = float(correct)/float(len(datapoints))
    return acc


attributelen = len(train_datapoints[1]) - 1
oneatttrees = oneDepthTree(attributelen)
twoatttrees = twoDepthTree(attributelen,oneatttrees)
threeatttrees = threeDepthTree(attributelen,oneatttrees,twoatttrees)

start_prob = float(1)/float(len(train_datapoints))

new_train = train_datapoints
for x in range(len(new_train)):
    new_train[x].append(start_prob)

ht,at = adaboost(10,new_train,threeatttrees)

for i in range(len(ht)):
    ht_ = ht[:(i+1)]
    at_ = at[:(i+1)]
    print("Number of rounds of boosting = "+str(i+1))
    acc_train = accuracy(ht_,at_,train_datapoints)
    print("for training data accuracy = "+str(acc_train))
    acc_test = accuracy(ht_,at_,test_datapoints)
    print("for testing data accuracy = "+str(acc_test))
    print(at_)








'''
Created on 20-Oct-2017

@author: Ashwin
'''
import csv


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
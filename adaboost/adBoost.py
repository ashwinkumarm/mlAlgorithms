'''
Created on 18-Oct-2017

@author: Ashwin
'''
'''
Created on 18-Oct-2017

@author: Ashwin
'''
import csv
from math import log, exp, sqrt

import pandas as pd
import queue


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


class Node:
    def __init__(self,value):
        self.val = value
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
            if x[temp.val] == "0":
                temp = temp.leftChild
            else:
                temp = temp.rightChild
            if temp == "0" or temp == "1":
                prediction = temp
                break
        return prediction

    def maxDepth(self, root):
        if root == None:
            return 0
        return max(self.maxDepth(root.leftChild),self.maxDepth(root.rightChild))+1
    
def constructHypothesisSpace(X,Y): 
    q = queue.Queue()
    m = len(X.columns)
    for i in range(m):   
        attr1 =  str(i)
        root = Node(attr1)
        root.leftChild = Node(0)
        q.put(root)
        root.leftChild = Node(1)
        q.put(root)
    
    for i in range(m):   
        attr1 = str(i)
        root = Node(attr1)
        root.rightChild = Node(0)
        q.put(root)
        root.rightChild = Node(1)
        q.put(root)
        
    for i in range(m):   
        attr1 =    str(i)
        root = Node(attr1)
        q.put(root)    
        
        
    while not q.empty():
        root = q.get()
        d = root.maxDepth(root)
        #print(d)
        if d == 1:
            for i in range(m):
                attr1 = str(i)
                node = Node(attr1)
                node.leftChild = Node(0)
                node.rightChild = Node(0)
                root.leftChild = node
                q.put(root)
                node.leftChild = Node(0)
                node.rightChild = Node(1)
                root.leftChild = node
                q.put(root)
                node.leftChild = Node(1)
                node.rightChild = Node(0)
                root.leftChild = node
                q.put(root)
                node.leftChild = Node(1)
                node.rightChild = Node(1)
                root.leftChild = node
                q.put(root)
        elif d == 2:
            for i in range(m):   
                attr1 =  str(i)
                node = Node(attr1)
                if root.rightChild == None:
                    node.leftChild = Node(0)
                    root.rightChild = node
                    q.put(root)
                    node.leftChild = Node(1)
                    root.rightChild = node
                    q.put(root)
                    node.righttChild = Node(0)
                    root.rightChild = node
                    q.put(root)
                    node.rightChild = Node(1)
                    root.rightChild = node
                    q.put(root)
                elif root.leftChild == None:
                    node.leftChild = Node(0)
                    root.leftChild = node
                    q.put(root)
                    node.leftChild = Node(1)
                    root.leftChild = node
                    q.put(root)
                    node.righttChild = Node(0)
                    root.leftChild = node
                    q.put(root)
                    node.rightChild = Node(1)
                    root.leftChild = node
                    q.put(root)

        elif d == 3 and root.rightChild == None:
            for i in range(m):   
                attr1 =  str(i)
                node = Node(attr1)
                node = Node(attr1)
                node.leftChild = Node(0)
                node.rightChild = Node(0)
                root.rightChild = node
                q.put(root)
                node.leftChild = Node(0)
                node.rightChild = Node(1)
                root.rightChild = node
                q.put(root)
                node.leftChild = Node(1)
                node.rightChild = Node(0)
                root.rightChild = node
                q.put(root)
                node.leftChild = Node(1)
                node.rightChild = Node(1)
                root.rightChild = node
        else:
            break        
    hyp = []    
    while not q.empty():
        n = q.get()
        hyp.append(n)
    return hyp    
 
def oneDepthTree(noOfAttributes):
    trees = []
    for i in range(noOfAttributes -1):
        for c1 in ["0","1"]:
            for c2 in ["0", "1"]:
                temp = Node(i+1)
                temp.leftChild = c1
                temp.rightChild = c2
                trees.append(temp)
    return trees


def twoDepthTree(attributelen,oneDepthTree):
    trees = []
    for i in range(attributelen -1):
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                temp = Node(i+1)
                temp.leftChild = c1
                temp.rightChild = one
                trees.append(temp)
        for one in oneDepthTree:
            for c1 in ["0","1"]:
                temp = Node(i+1)
                temp.leftChild = one
                temp.rightChild = c1
                trees.append(temp)
    return trees

def threeDepthTree(attributelen,oneDepthTree,twoDepthTree):
    trees = []
    for i in range(attributelen -1):
        for one in oneDepthTree:
            for two in oneDepthTree:
                temp = Node(i+1)
                temp.leftChild = one
                temp.rightChild = two
                trees.append(temp)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                temp = Node(i+1)
                temp.leftChild = c1
                temp.rightChild = one
                trees.append(temp)
        for one in twoDepthTree:
            for c1 in ["0","1"]:
                temp = Node(i+1)
                temp.leftChild = one
                temp.rightChild = c1
                trees.append(temp)
    return trees


def adaboost(M,train_datapoints,threeDepthTree):
    ht = []
    at = []
    for i in range(M):
        besttree = None
        besterr = 1
        errindexes = []
        for tree in threeDepthTree:
            err,indexes = tree.error(train_datapoints)
            if err < besterr:
                besterr = err
                besttree = tree
                errindexes = indexes
        a = float(1/2) * log((1-besterr)/besterr)
        ht.append(besttree)
        at.append(a)
        for j in range(len(train_datapoints)):
            point = train_datapoints[j]
            w_old = point[len(point) - 1]
            w_new = 1
            if j in errindexes:
                w_new = (w_old * exp(a))/(2 * sqrt((1 - besterr) * besterr))
            else:
                w_new = (w_old * exp(-1 * a)) / (2 * sqrt((1 - besterr) * besterr))
            train_datapoints[j][len(point) - 1] = w_new
        # print (float(1) - besterr)
    return (ht,at)


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


#attributelen = len(train_datapoints[1]) - 1
#oneatttrees = oneDepthTree(attributelen)
#twoatttrees = twoDepthTree(attributelen,oneatttrees)
#threeatttrees = threeDepthTree(attributelen,oneatttrees,twoatttrees)

df = pd.read_csv("D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data",sep = ',',index_col = False, header = None, names = ["22","0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"])
X = df[["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]]  
Y = df[['22']]
    
start_prob = float(1)/float(len(train_datapoints))

new_train = train_datapoints
for x in range(len(new_train)):
    new_train[x].append(start_prob)

hyp = constructHypothesisSpace(X,Y)
ht,at = adaboost(4,new_train,hyp)

for i in range(len(ht)):
    ht_ = ht[:(i+1)]
    at_ = at[:(i+1)]
    print("Number of rounds of boosting = "+str(i+1))
    acc_train = accuracy(ht_,at_,train_datapoints)
    print("for training data accuracy = "+str(acc_train))
    acc_test = accuracy(ht_,at_,test_datapoints)
    print("for testing data accuracy = "+str(acc_test))








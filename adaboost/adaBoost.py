'''
Created on 17-Oct-2017

@author: Ashwin
'''

import time

import pandas as pd
import queue


class Node:
    def __init__(self,i):
        self.val = i
        self.leftChild = None
        self.rightChild = None

    def maxDepth(self, root):
        if root == None:
            return 0
        return max(self.maxDepth(root.leftChild),self.maxDepth(root.rightChild))+1
        
def majorityCount(X,Y,attr1 = None,attr1_val = None,attr2 = None,attr2_val = None,attr3 = None,attr3_val=None):
    if attr2_val != None and attr3_val != None:
        a =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val) & (X[attr3] == attr3_val) & (Y['22'] == 1)])
        b =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val) & (X[attr3] == attr3_val) & (Y['22'] == 0)])
    elif attr2_val != None:
        a =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val)  & (Y['22'] == 1)])
        b =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val)  & (Y['22'] == 0)])
    else:
        a =len(X[(X[attr1] == attr1_val ) & (Y['22'] == 1)])
        b =len(X[(X[attr1] == attr1_val ) & (Y['22'] == 0)])
     
    if a > b:
        return Node(str(1))
    else:
        return Node(str(0))   
    
def constructHypothesisSpace(X,Y): 
    q = queue.Queue()
    m = len(X.columns)
    for i in range(m):   
        attr1 = "X" + str(i)
        root = Node(attr1)
        root.leftChild = Node(0)
        q.put(root)
        root.leftChild = Node(1)
        q.put(root)
    
    for i in range(m):   
        attr1 = "X" + str(i)
        root = Node(attr1)
        root.rightChild = Node(0)
        q.put(root)
        root.rightChild = Node(1)
        q.put(root)
        
    for i in range(m):   
        attr1 = "X" + str(i)
        root = Node(attr1)
        q.put(root)    
        
        
    while not q.empty():
        root = q.get()
        d = root.maxDepth(root)
        #print(d)
        if d == 1:
            for i in range(m):
                attr1 = "X" + str(i)
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
                attr1 = "X" + str(i)
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
                attr1 = "X" + str(i)
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

      
def formHypothesisSpace(X,Y):
    hyp = []
    m = len(X.columns)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                attr1 = "X" + str(i)
                attr2 = "X" + str(j)
                attr3 = "X" + str(k)
                h1 = Node(attr1)
                node2 = Node(attr2)
                node3 = Node(attr3)
                node3.leftChild = majorityCount(X,Y, str(i), 1, str(j), 1, str(k), 0)
                node3.rightChild = majorityCount(X,Y, str(i), 1, str(j), 1, str(k), 1)
                node2.leftChild = majorityCount(X,Y,str(i), 1, str(j), 0, str(k), None)
                node2.rightChild = node3
                h1.leftChild = majorityCount(X,Y,str(i), 0, str(j), None, str(k), None)
                h1.rightChild = node2
                hyp.append(h1)
                h2 = Node(attr1)
                node3.leftChild = majorityCount(X,Y,str(i), 0, str(j), 0, str(k), 0)
                node3.rightChild = majorityCount(X,Y,str(i), 0, str(j), 0, str(k), 1)
                node2.leftChild = node3
                node2.rightChild = majorityCount(X,Y,str(i), 0, str(j), 1, str(k), None)
                h2.leftChild = node2
                h2.rightChild = majorityCount(X,Y,str(i), 1, str(j), None, str(k), None)
                hyp.append(h2)
                h3 = Node(attr1)
                node2.leftChild = majorityCount(X,Y,str(i), 0, str(j), 0, str(k), None)
                node2.rightChild = majorityCount(X,Y,str(i), 0, str(j), 1, str(k), None)
                node3.leftChild = majorityCount(X,Y,str(i), 1, str(j), 0, str(k), None)
                node3.rightChild = majorityCount(X,Y,str(i), 1, str(j), 1, str(k), None)
                h3.leftChild = node2
                h3.rightChild = node3
                hyp.append(h3)
       
    print(len(hyp))            
      
    
def adaboost():
    df = pd.read_csv("D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data",sep = ',',index_col = False, header = None, names = ["22","0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"])
    X = df[["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21"]]  
    Y = df[['22']]
    m = len(X.columns)
    #print(m)
    attr1 = '3'
    attr2 = '1'
    attr3 = '2'
    attr1_val = 1
    attr2_val = 0
    attr3_val = 1
    #n = majorityCount(attr1, attr1_val, attr2, attr2_val, attr3, attr3_val)
    #print(n.val)
    #a =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val) & (X[attr3] == attr3_val) & (Y['22'] == 1)])
    #b =len(X[(X[attr1] == attr1_val ) & (X[attr2] == attr2_val) & (X[attr3] == attr3_val) & (Y['22'] == 0)])
    #print(a)
    #print(b)
    #print(X[X[(a)] == 1])
    
    #start = time.clock()
    #formHypothesisSpace(X,Y)
    #print(time.clock() - start)
    #majorityCount(X,Y,attr1, attr1_val, attr2, attr2_val, attr3, attr3_val)
    #constructHypothesisSpace(X, Y)
    attr1 = "X" + str(1)
    root = Node(attr1)    
    print(root.maxDepth(root))            
 

adaboost()          
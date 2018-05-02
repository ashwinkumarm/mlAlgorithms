'''
Created on 02-Sep-2017

@author: Ashwin
'''

import numpy as np
import pandas as pd


def perceptron():
    df = pd.read_csv("D:\semester 3\machine learning\Assgn\Assgn1\perceptron.data",sep = ',',header = None, names = ["A","B","C","D","E"])
    X = df[['A','B','C','D']]  ## 1000 X 4
    Y = df[['E']] #1000 X 1
    #print(X.values)
    #X = X.T
    #print(X)
    #print(X[1:])
    #print(X.head(5))
    #print(Y.head(5))
    w = np.zeros(4)    ## 1X4
    #print(w.dot(w.transpose()))
    #print(w.T)
    #print(w)
    #print(w.describe) 
    b= 0
    #print(w)
    #r1 = X.loc()
    #for idx,row in X.iterrows():
     #   print(row.values)
    #for z in X.iterrows():
        #print(z[1])
    #print(r1)
    c = True
    iteration = 0;
    while(c):
        iteration+=1;
        original_w = w
        original_b = b
        w = original_w + 100000000000  * updateW(original_w,original_b,X,Y)
        b = original_b + 100000000000 * updateB(original_w,original_b,X,Y)
        if np.array_equal(original_w,w):
            if np.array_equal(original_b,b):
                c = False
        print(w)
        print(b)
        #print(w)
    print(iteration)    
    return w

def updateW(w, b, X,Y):
    wsum = np.zeros(4)
    for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):   ## x-1X4   y-1X1 
        y = y.item()  
        q =  (-y) * ((x.values.dot(w.transpose())) + b)
        #q =  x.values.dot(w.transpose()) 
        #q = q + b
        #y = y.item()
        #q = (-y) * q
        if q >= 0:
            #print(y.shape)
            wsum += np.dot(x.values, y)
    return wsum       
        
def updateB(w, b, X, Y):
    bsum = 0
    for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):  ## x-1X4   y-1X1 
        y = y.item()  
        q =  (-y) * ((x.values.dot(w.transpose())) + b) 
        #q = q + b
        #q = (-y) * qv
        if q >= 0:
            #print(y.shape)
            bsum += y 
    return bsum  
    
            
perceptron()        
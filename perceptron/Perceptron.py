'''
Created on 10-Sep-2017

@author: Ashwin
'''

import numpy as np
import pandas as pd


def perceptron():
    df = pd.read_csv("D:\semester 3\machine learning\Assgn\Assgn1\perceptron.data",sep = ',',header = None, names = ["A","B","C","D","E"])
    X = df[['A','B','C','D']]  ## 1000 X 4
    Y = df[['E']] #1000 X 1
    w = np.zeros(4)    ## 1X4
    b= 0
    c = True
    alpha = 0
    iteration = 0;
    while(c):
        iteration+=1;
        original_w = w
        original_b = b
        w = original_w + alpha  * updateW(original_w,original_b,X,Y)
        b = original_b + alpha * updateB(original_w,original_b,X,Y)
        if np.array_equal(original_w,w):
            if np.array_equal(original_b,b):
                c = False
        print(w)
        print(b)
    print(iteration-1)    
    return w

def updateW(w, b, X,Y):
    wsum = np.zeros(4)
    for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):   ## x-1X4   y-1X1 
        y = y.item()  
        q =  (-y) * ((x.values.dot(w.transpose())) + b)
        if q >= 0:
            wsum += np.dot(x.values, y)
    return wsum       
        
def updateB(w, b, X, Y):
    bsum = 0
    for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):  ## x-1X4   y-1X1 
        y = y.item()  
        q =  (-y) * ((x.values.dot(w.transpose())) + b) 
        if q >= 0:
            bsum += y 
    return bsum  
    
            
perceptron()        
'''
Created on 10-Sep-2017

@author: Ashwin
'''

import pandas as pd
import numpy as np


def perceptron():
    df = pd.read_csv("perceptron.data",sep = ',',header = None, names = ["A","B","C","D","E"])
    X = df[['A','B','C','D']]  
    Y = df[['E']] 
    w = np.zeros(4)
    b= 0
    c = True
    alpha = 1        
    iteration = 0;
    while(c):
        iteration+=1;
        original_w = w
        original_b = b
        w,b = updateWAndB(original_w,original_b,X,Y)
        w = original_w + alpha  * w
        b = original_b + alpha * b
        if np.array_equal(original_w,w):
            if np.array_equal(original_b,b):
                c = False
        print(w)
        print(b)
    print(iteration-1)    
    return w

def updateWAndB(w, b, X,Y):
    wsum = np.zeros(4)
    bsum = 0
    for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):    
        y = y.item()  
        q =  (-y) * ((x.values.dot(w.transpose())) + b)
        if q >= 0:
            wsum += np.dot(x.values, y)
            bsum += y 
    return wsum,bsum       
        
    
            
perceptron()        

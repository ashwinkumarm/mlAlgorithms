'''
Created on 10-Sep-2017

@author: Ashwin
'''
import numpy as np
import pandas as pd


def perceptron():
    df = pd.read_csv("perceptron.data",sep = ',',header = None, names = ["A","B","C","D","E"])
    X = df[['A','B','C','D']]  
    Y = df[['E']] 
    w = np.zeros(4)
    b= 0
    c = True
    iteration = 0;
    alpha = 1
    while(c):
        r= 0
        literation = 0
        for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):   
            r +=1  
            y = y.item()  
            q =  (-y) * ((x.values.dot(w.transpose())) + b)
            if q >= 0:
                iteration += 1   
                literation+=1
                prevW = w
                prevB = b
                w = w + alpha * np.dot(x.values, y)
                b = b + alpha * y
            if np.array_equal(prevW,w) and np.array_equal(prevB,b):
                c = False
                print(w)
                break       
            # if all the points are correctly classified
            if r == 1000 and literation== 0:
                print(w)
                c = False
                break
        
    print(iteration-1)  
          
perceptron()   

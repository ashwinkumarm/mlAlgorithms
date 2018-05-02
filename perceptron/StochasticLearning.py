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
    iteration = 0;
    alpha = 1
    while(c):
        print("----------------------------------------------")
        r= 0
        literation = 0
        for (idx,x),(idy,y) in zip(X.iterrows(),Y.iterrows()):   
            r +=1 ## x-1X4   y-1X1 
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
                break       
            print(w)
            #print(prevW)
            print(b)
            if r == 1000 and literation== 0:
                print(w)
                c = False
                break
        
    print(iteration-1)  
          
perceptron()   

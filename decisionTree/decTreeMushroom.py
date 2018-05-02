'''
Created on 01-Oct-2017

@author: Ashwin
'''

import numpy as np
import pandas as pd


def readData():
    df = pd.read_csv("D:\semester_3\machine learning\Assgn\Assgn2\mush_train.data", header = None)
    X = df.iloc[:,1:23]
    Y = df.iloc[:,0]    
    return X,Y

def findEntropy(X,Y):
    noOfAttr = len(X.columns)
    print(noOfAttr)
    for attr in range(noOfAttr):
        findIG(attr)

def findIG(attr):
    print("")
    
[X, Y] = readData()    
findEntropy(X, Y)
#print(X)
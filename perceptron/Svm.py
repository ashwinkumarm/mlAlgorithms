'''
Created on 07-Sep-2017

@author: Ashwin
'''

import numpy as np
import pandas as pd


def colvec(rowvec):
    v = np.asarray(rowvec)
    return v.reshape(v.size,1)

def SVM():
    df = pd.read_csv("D:\semester 3\machine learning\Assgn\Assgn1\mystery.data",sep = "," ,header = None, names = ['A','B','C','D','E'])
    X = df[['A','B','C','D']]
    Y = df[['E']]
    w = np.zeros(4)    ## 1X4
    w_t = colvec(w)
    b= 0
    #print(w_t)
    x = np.vstack((w_t,b))
    h = np.vstack((np.eye(4),np.zeros(4)))
    f = np.zeros(5)
    print(h)
    
SVM()    
'''
Created on 08-Sep-2017

@author: Ashwin
'''
from sklearn import svm

import numpy as np


lines = [line.strip('\n') for line in open('D:\semester 3\machine learning\Assgn\Assgn1\mystery.data','r')]
linelist = np.empty((1000, 5), dtype=float) 
x=[]
y=[]
fignum = 1    
for i in range(0,1000):
    j=0
    for s in lines[i].split(','):
        linelist[i][j]=float(s)
        j=j+1
    x.append(np.array(linelist[i][0:4]))
    y.append(linelist[i][4])
    
clf = svm.SVC(kernel='linear', C =0.39)
clf.fit(x,y)
w=clf.coef_[0]
print("The optimized weight vector is "+str(w))
b=y[clf.support_[0]]-(x[clf.support_[0]]*(np.matrix(w).transpose())) 
print("The value of bias is "+str(b))
margin = 1 / np.sqrt((w ** 2).sum())
print("The optimal margin is "+str(margin))

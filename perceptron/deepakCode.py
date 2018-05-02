'''
Created on 03-Sep-2017

@author: Ashwin
'''
import numpy as np


lines = [line.strip('\n') for line in open("D:\semester 3\machine learning\Assgn\Assgn1\perceptron.data",'r')]
linelist = np.empty((100, 5)) 
x=[]
w=[]
b=[]   
for i in range(0,100):
    j=0
    for s in lines[i].split(','):
        linelist[i][j]= s
        j=j+1
    x.append(np.matrix(np.array(linelist[i][0:4])))
        
w.append(np.matrix(np.zeros((1,4),dtype=np.int)))
b.append(0)
losslist=[]
loss=[]
for i in range(0,100):
    func=(x[i]*w[0].transpose())+b[0]
    temp=-linelist[i][4]*func
    if temp>=0:
        losslist.append(i)

sum_w=np.matrix(np.zeros((1,4),dtype=np.float64))
sum_b=0
k=1
count=0
j=0
print("The values of w and b:")
while count<=20 and losslist:
    for j in losslist:
        i=j
        sum_w+=np.dot(linelist[i][4],x[i])
        sum_b+=linelist[i][4]
    w.append(w[k-1]+(2*sum_w))
    b.append(b[k-1]+(2*sum_b))
    losslist[:]=[]
    print(w[k])
    print(b[k])
    for m in range(0,100):
        func=(x[m]*w[k].transpose())+b[k]
        temp=-linelist[m][4]*func
        if temp>=0:
            losslist.append(m)
    k+=1
    sum_w=0
    sum_b=0
    count+=1
print("The number of iterations are "+str(count))


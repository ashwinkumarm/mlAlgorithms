'''
Created on 10-Sep-2017

@author: Ashwin
'''
import numpy as np


lines = [line.strip('\n') for line in open('D:\semester 3\machine learning\Assgn\Assgn1\perceptron.data','r')]
linelist = np.empty((100, 5))
x=[]
w=[]
b=[]   
for i in range(0,100):
    j=0
    for s in lines[i].split(','):
        linelist[i][j]=s
        j=j+1
    x.append(np.matrix(np.array(linelist[i][0:4])))
    
w.append(np.matrix(np.zeros((1,4),dtype=np.int)))
b.append(0)
losslist=[]
k=0
count=0
iteration=0
while count<=2000:
    losslist[:]=[]
    for i in range(0,100):
        func=(x[i]*w[k].transpose())+b[k]
        temp=-linelist[i][4]*func
        if temp>=0:
            losslist.append(i)
    if not losslist:
        break
    
    func=(x[iteration]*w[k].transpose())+b[k]
    temp=-linelist[iteration][4]*func
    if temp>=0:
        sum_w=np.dot(linelist[iteration][4],x[iteration])
        sum_b=linelist[iteration][4]
        w.append(w[k]+(1*sum_w))
        b.append(b[k]+(1*sum_b))
        k+=1
        print(w[k])
        print(b[k])
    else:
        w.append(w[k])
        b.append(b[k]) 
        k+=1
        print(w[k])
        print(b[k])
    count+=1
    iteration=(iteration+1)%len(linelist)
print("The number of iterations are "+str(count))

'''
Created on 18-Oct-2017

@author: Ashwin
'''
import csv
import sys

import numpy as np


fvlist=list()
fvlist1=list()
fnlist=list()
hspace=list()
chosen_classifiers=list()

class Node:
    def __init__(self, fname=None, flist=None):
        if fname is None:
            self.fname=''
        else:
            self.fname=fname
        if flist is None:
            self.flist=[]
        else:
            self.flist=flist
        self.children=[]

    def get_fname(self):
        return self.fname
    
    def get_flist(self):
        return self.flist
    
def read_train(input_data):
    with open(input_data) as inp:
        reader=csv.reader(inp)
        for row in reader:
            temp = (row[1:],row[0])
            fvlist.append(temp)
        for x in range(1,len(fvlist[0][0])+1):
            name='F'+str(x)
            fnlist.append(name)
    return

def read_test(input_data):
    with open(input_data) as inp:
        reader=csv.reader(inp)
        for row in reader:
            temp = (row[1:],row[0])
            fvlist1.append(temp)
    return

def hypothesis_space():
    for i in range(0,len(fnlist)):   
        for j in range(2):
            for k in range(2):
                class_value=[]
                class_value.insert(0,str(j))
                class_value.insert(1,str(k)) 
                root=Node(fnlist[i],class_value)    
                hspace.append(root)   
    return

def loss(data,index,alpha):
    total=0
    for i in range(len(hspace)):
        if i!=index:
            node=hspace[i]
            flist=node.get_flist()
            root_value=data[0][fnlist.value(node.get_fname())]
            if int(flist[int(root_value)])==0:
                total-=alpha[i]
            else:
                total+=alpha[i]
    if int(data[1])==1:
        total*=-1
    return np.exp(total)

def analyse(data,tree):  
        flist=tree.get_flist()
        parent_value=data[0][fnlist.value(tree.get_fname())]
        if int(flist[int(parent_value)])!=int(data[1]):
            return 0
        else:
            return 1

def coorddesc():
    alpha=[0]*len(hspace)
    final_hyp=[False]*len(hspace)
    count=0
    while True:
        for i in range(len(hspace)):
            if ~final_hyp[i]:
                num=0
                den=0
                for data in fvlist:
                    if analyse(data,hspace[i])==1:
                        num+=loss(data,i,alpha)
                    else:
                        den+=loss(data,i,alpha)
                temp=(1/2)*(np.log(num/den))
                if temp==alpha[i]:
                    final_hyp[i]=True
                alpha[i]=temp
        count+=1
        if set(final_hyp) and len(set(final_hyp))==1:
            break
        if count==200:
            break
    for i in range(len(hspace)):
        chosen_classifiers.append((hspace[i],alpha[i]))
    print('Hypothesis Weight: ',alpha)
    exponential_loss = 0
    for data in fvlist:
        exponential_loss+=loss(data,-1,alpha)
    print('Exponential Loss: ',exponential_loss)

def accuracy():
    correct=0
    for i in range(len(fvlist1)):
        total=0
        for h in chosen_classifiers:
            total+=accuracy_parse(fvlist1[i],h[0])*h[1]
        if(total>0 and int(fvlist1[i][1])==1)or(total<=0 and int(fvlist1[i][1])==0):
            correct+=1
    print('Accuracy of Coordinate Descent Algorithm on Test Data',float(correct)/float(len(fvlist1))*100)


def accuracy_parse(data,tree):
        flist=tree.get_flist()
        root_value=data[0][fnlist.value(tree.get_fname())]
        if int(root_value)==0:
            if int(flist[0])==0:
                return -1
            else:
                return 1
        else:
            if int(flist[1])==0:
                return -1
            else:
                return 1

def main():
    train_data="D:\semester_3\machine learning\Assgn\Assgn3\heart_train.data"
    test_data="D:\semester_3\machine learning\Assgn\Assgn3\heart_test.data"
    read_train(train_data)
    read_test(test_data)
    hypothesis_space()
    coorddesc()
    accuracy()
    
main()

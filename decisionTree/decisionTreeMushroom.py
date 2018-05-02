'''
Created on 30-Sep-2017

@author: Ashwin
'''
import csv
from math import log

def calculateEntropy(C, x_only,u,entropy):
    count = float(x_only.count(u))
    noOfElements = [x for x in C if x[1] == u]
    p = [x[0] for x in noOfElements if x[0] == '+']
    e = [x[0] for x in noOfElements if x[0] == '-']
    np = float(len(p))
    ne = float(len(e))
    npcount = np/count
    necount = ne/count

    if np == 0:
        nplog = 0
    else:
        nplog = npcount * log(npcount,2)

    if ne == 0:
        nelog = 0
    else:
        nelog = necount * log(necount,2)

    internalSum = nplog + nelog

    entropy = entropy + (-count/float(len(C)))* internalSum
    return entropy

class TreeNode:
    def __init__(self,i):
        self.value = i
        self.children = {}
            
def formTree(data):
    parameter = None
    maxEntropy = None
    Y = [x[0] for x in data]
    count_p = Y.count('+')
    count_e = Y.count('-')

    if count_p == 0:
        return '-'
    if count_e == 0:
        return '+'

    for i in range(len(data[0])-1):
        pair = [[x[0],x[i+1]] for x in data]
        param = [x[1] for x in pair]
        unique = list(set(param))
        entropy = 0
        for u in unique:
            entropy = calculateEntropy(pair, param, u, entropy)

        if maxEntropy is None:
            maxEntropy = entropy
            parameter = i+1
        else:
            if entropy <= maxEntropy:
                maxEntropy = entropy
                parameter = i + 1

    root = TreeNode(parameter)
    parameterElements = [x[parameter] for x in data]
    uniqueParameterElements = list(set(parameterElements))
    for UC in uniqueParameterElements:
        childParam = [x for x in data if x[parameter] == UC]
        childNode = formTree(childParam)
        root.children[UC] = childNode

    return root


def predict(root = TreeNode,datapoint = None):
    index = root.value
    child = root.children.get(datapoint[index])
    if child is None:
        return 'z'
    if child == '+' or child == '-':
        return child
    else:
        return predict(child,datapoint)

def accuracy(root = TreeNode,datapoints = [[]]):
    correct = 0
    for datapoint in datapoints:
        prediction = predict(root,datapoint)
        if prediction == datapoint[0]:
            correct = correct +1
    percent_correct = float(correct)/float(len(datapoints))
    print(percent_correct)
    
    
trainData = []
fileO = open("D:/semester_3/bigData/Assignment/Assgn4/decisionTreeInp.txt", "r")
with fileO as f:
    content = f.readlines()
    trainData.append([x for x in content])
#filereader = csv.reader(File)
fileO.close()

# for row in File:
#     trainData.append([x for x in row])


# File = open("D:\semester_3\machine learning\Assgn\Assgn2\mush_test.data")
# filereader = csv.reader(File)
# testData = []
# for row in filereader:
#     testData.append([x for x in row])
# File.close()    
    
root = formTree(trainData)

print("Training Data accuracy is")
accuracy(root,trainData)
# print("Test data accuracy is")
# accuracy(root,testData)


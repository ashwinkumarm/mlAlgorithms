'''
Created on 10-Sep-2017

@author: Ashwin
'''
from decimal import Decimal


#inital points
# w1x1 + w2x2+ w3x3 + x4x4 + b = 0 linear separator
w1 = 0
w2 = 0
w3 = 0
w4 = 0
b = 0

stepsize = 1

def linearseparator(dataList):
    deltaw1 = 0
    deltaw2 = 0
    deltaw3 = 0
    deltaw4 = 0
    deltab = 0
    for datapoint in dataList:
        yprediction =  w1*datapoint[0] + w2 * datapoint[1] + w3 * datapoint[2] + w4 * datapoint[3] + b
        if yprediction * datapoint[4] <= 0 :
            deltaw1 = deltaw1 + datapoint[0] * datapoint[4]
            deltaw2 = deltaw2 + datapoint[1] * datapoint[4]
            deltaw3 = deltaw3 + datapoint[2] * datapoint[4]
            deltaw4 = deltaw4 + datapoint[3] * datapoint[4]
            deltab = deltab + datapoint[4]
            return (deltaw1,deltaw2,deltaw3,deltaw4,deltab)
    return (deltaw1,deltaw2,deltaw3,deltaw4,deltab)

datalist = []

with open("D:\semester 3\machine learning\Assgn\Assgn1\perceptron.data") as f:
    content = f.readlines()

for line in content:
    datapoint = []
    datastring = line.split(",")
    for data in datastring:
        datapoint.append(float(data))
    datalist.append(datapoint)

iterations = 0
while True:
    iterations = iterations+1
    delta = linearseparator(datalist)
    if delta == (0,0,0,0,0) :
        iterations = iterations-1
        print("Number of iterations "+str(iterations))
        break
    w1 = w1 + stepsize * delta[0]
    w2 = w2 + stepsize * delta[1]
    w3 = w3 + stepsize * delta[2]
    w4 = w4 + stepsize * delta[3]
    b = b + stepsize * delta[4]
    
print (w1,w2,w3,w4,b)
print("done")  
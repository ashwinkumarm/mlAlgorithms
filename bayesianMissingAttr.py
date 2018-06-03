'''
Created on 26-Nov-2017

@author: Ashwin
'''
import numpy as np
from math import log
import sys



def missingProb(data, column):
    m = data.shape[0]
    missCount = 0
    for d in data:
        if d[column] == '?':
            missCount = missCount +1
    return float(missCount)/m

def MissingDataCombination(ui,points):
    m = ui[0]
    new_points = []
    for p in points:
        p[m] = 'y'
        new_points.append(p)
        p[m] = 'n'
        new_points.append(p)
    if len(ui) == 1:
        return new_points
    return MissingDataCombination(ui[1:],new_points)


def updatetheta(data,theta,edges):
    updatedThetha = []
    for e in edges:
        if e[1] is None:
            allValues = data[:,e[0]]
            valuesWithY = np.where(allValues == 'y')[0]
            updatedThetha.append(float(len(valuesWithY))/float(len(allValues)))
        else:
            allValues = data[:,list(e)]
            valuesWithY = allValues[allValues[:,0] == 'y']
            apn = (float(len(valuesWithY)) / float(len(allValues)))
            updatedThetha.append(apn)
    return updatedThetha


def logProbSum(point,theta,edges):
    logSum = 0
    for i in range(len(point)):
        f = point[i]
        p = theta[i]
        if f != 'y':
            p = 1 - p
        logSum = logSum + log(p)
    return logSum

def probabilitySummation(unKnownIndex,knownIndex,point,data):
    knownData = data[:,knownIndex]
    ni_point = point[knownIndex]
    ncount = 0
    ucount = 0
    for i in range(len(knownData)):
        d = knownData[i]
        if np.array_equal(d ,ni_point ):
            ncount = ncount + 1
            unknownDat = data[i][unKnownIndex]
            un_point = point[unKnownIndex]
            if np.array_equal(unknownDat, un_point):
                ucount = ucount + 1
    return float(ucount)/float(ncount)

def newPoint(point, theta, data,edges):
    ui = np.where(point == '?')[0]
    ni = np.where(point != '?')[0]
    new_point = None
    sum_prob = 0
    if len(ui) == 0:
        new_point = point
        sum_prob = 0
    else:
        slist = []
        slist.append(point)
        comb = MissingDataCombination(ui,slist)
        new_point = None
        max_prob = -1 * sys.maxint
        sum_prob = 0
        for c in comb:
                logSUm = logProbSum(c,theta,edges)
                summat = probabilitySummation(ui,ni,c,data)
                cp = summat * logSUm
                sum_prob = sum_prob + cp
                if cp >= max_prob:
                    new_point = c
                    max_prob = cp

    return np.array(new_point),sum_prob    
    



def bayesianMain():
    edges = [(0,None),(1,12),(2,13),(3,8),(4,0),(5,4),(6,5),(7,8),(8,5),(9,5),(10,2),(11,0),(12,5),(13,5),(14,4),(15,5),(16,7)]
    theta = np.random.rand(len(edges))
    data = np.genfromtxt("D:\semester_3\machine learning\Assgn\Assgn5\congress.data", delimiter=',', dtype=str)
    #theta = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    prevProb = -1 * sys.maxint
    iteration = 1
    
    print "missing Probability for each attributes are:"
    
    missProb = []
    for col in range(17):
        missProb.append(missingProb(data, col))
    
    print missProb
    
    while True:
        data = np.genfromtxt("D:\semester_3\machine learning\Assgn\Assgn5\congress.data", delimiter=',', dtype=str)
        print "Thetha values in iteration no :"+str(iteration)
        prob = 0
        dataWithMissingValue = []
       
        print theta
    
        for d in data:
            completePoint,jointProb = newPoint(d,theta,data,edges)
            prob = prob + jointProb
            dataWithMissingValue.append(np.array(completePoint))
        
        print "Log-Likelihood value is:"    
        print prob
        
        if prob - prevProb < 0.0001:
            break
        prevProb = prob
        theta = np.array(updatetheta(np.array(dataWithMissingValue),theta,edges))
        iteration = iteration + 1
        
     
        
bayesianMain()        
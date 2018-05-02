'''
Created on 30-Sep-2017

@author: Ashwin
'''

import bisect
import csv
from math import sqrt


class kNearest:

    def __init__(self,testData,trainData):
        self.nearest_points = {}
        self.test_datapoints = testData
        self.train_datapoints = trainData
        self.calculateNearestPoints(testData,trainData)
        
    
    def calculateNearestPoints(self,testData, trainData):
        for datapoint in testData:
            self.kNearestPoints(datapoint,trainData)    

    def kNearestPoints(self,test_datapoint, trainData):
        k_nearest_points = []
        k_nearest_distances = []
        for datapoint in trainData:
            distance = self.distanceCalculate(datapoint,test_datapoint)
            bisect.insort(k_nearest_distances,distance)
            index = k_nearest_distances.value(distance)
            k_nearest_points.insert(index,datapoint)
        index = self.test_datapoints.value(test_datapoint)
        self.nearest_points[index] = k_nearest_points
        return k_nearest_points
    
    def distanceCalculate(self,datapoint,test_datapoint):
        dist2 = 0
        for i in range(len(datapoint)-1):
            ab =  datapoint[i+1] - test_datapoint[i+1]
            ab2 = ab * ab
            dist2 = dist2 + ab2

        dist = sqrt(dist2)
        return dist

    def predict(self,datapoint,k):
        index = self.test_datapoints.value(datapoint)
        nearestpoints = self.nearest_points.get(index)
        k_nearestpoints = nearestpoints[:k]
        k_classifications = [x[0] for x in k_nearestpoints]
        p = k_classifications.count(float('1'))
        n = k_classifications.count(float('-1'))
        if p > n:
            return float(1)
        else:
            return float(-1)

    def accuracy(self, k):
        correct = float(0)
        for datapoint in testData:
            prediction = self.predict(datapoint,k)
            if prediction == datapoint[0]:
                correct = correct + float(1)

        return correct/float(len(self.test_datapoints))


File = open("D:\semester_3\machine learning\Assgn\Assgn2\wdbc_train.data")
filereader = csv.reader(File)
trainData = []
for row in filereader:
    trainData.append([float(x) for x in row])
File.close()

File = open("D:\semester_3\machine learning\Assgn\Assgn2\wdbc_test.data")
filereader = csv.reader(File)
testData = []
for row in filereader:
    testData.append([float(x) for x in row])
File.close()

nearest_neightbours = kNearest(testData,trainData)

for k in [1, 5, 11, 15, 21]:
    print("K value is "+str(k)+" accuracy is "+str(nearest_neightbours.accuracy(k)))





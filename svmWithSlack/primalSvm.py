'''
Created on 29-Sep-2017

@author: Ashwin
'''
import csv

from sklearn import svm


File = open("wdbc_train.data")
reader = csv.reader(File)
datapoints = []
for row in reader:
    datapoints.append([float(x) for x in row])
File.close()

X_train = [x[1:] for x in datapoints]
y_train = [x[0] for x in datapoints]

def predict(c,X_train,y_train,X_test,y_test):
    clf = svm.LinearSVC(C=c,dual=False)
    clf.fit(X_train, y_train)
    print(c, clf.accuracy(X_test,y_test))


File = open("wdbc_valid.data")
reader = csv.reader(File)
datapoints = []
for row in reader:
    datapoints.append([float(x) for x in row])
File.close()

X_valid = [x[1:] for x in datapoints]
y_valid = [x[0] for x in datapoints]

predict(1,X_train,y_train,X_train,y_train)
predict(10,X_train,y_train,X_train,y_train)
predict(100,X_train,y_train,X_train,y_train)
predict(1000,X_train,y_train,X_train,y_train)
predict(10000,X_train,y_train,X_train,y_train)
predict(100000,X_train,y_train,X_train,y_train)
predict(1000000,X_train,y_train,X_train,y_train)
predict(10000000,X_train,y_train,X_train,y_train)
predict(100000000,X_train,y_train,X_train,y_train)


print("for validation set")

predict(1,X_train,y_train,X_valid,y_valid)
predict(10,X_train,y_train,X_valid,y_valid)
predict(100,X_train,y_train,X_valid,y_valid)
predict(1000,X_train,y_train,X_valid,y_valid)
predict(10000,X_train,y_train,X_valid,y_valid)
predict(100000,X_train,y_train,X_valid,y_valid)
predict(1000000,X_train,y_train,X_valid,y_valid)
predict(10000000,X_train,y_train,X_valid,y_valid)
predict(100000000,X_train,y_train,X_valid,y_valid)

print("selected C is 10")

File = open("wdbc_test.data")
reader = csv.reader(File)
datapoints = []
for row in reader:
    datapoints.append([float(x) for x in row])
File.close()

X_test = [x[1:] for x in datapoints]
y_test = [x[0] for x in datapoints]

predict(10,X_train,y_train,X_test,y_test)
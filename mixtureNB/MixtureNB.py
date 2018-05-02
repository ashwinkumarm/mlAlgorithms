'''
Created on 12-Nov-2017

@author: Ashwin
'''
from naive import *
from math import log
import sys
from random import uniform
import numpy as np


class MixtureNB:

    def _init_(self):

        self.int = None
        self.unique = None
        self.X = None
        self.Y = None
        self.yprobs = {}
        
    def acc(self,X,Y):
        c = 0
        for i in range(len(X)):
            x = X[i]
            pred = self.prediction(x)
            if pred == Y[i]:
                c = c + 1
        return float(c)/float(len(X))

    def guess(self,X,p,inp):
        pred = p
        for i in range(len(X)):
            try:
                x = X[i]
                iFinal = inp[:,i]
                lx = len(iFinal[ iFinal == x ])
                distr = float(lx)/float(len(inp))
                if distr == 0:
                    distr = .00000000000001
                pred = pred * distr
            except:
                print inp
        return pred    

    def prediction(self,X):
        pre = 0
        s_y = None
        xs = self.X
        Y = self.Y
        XY = np.append(xs, Y.reshape(len(Y), 1), 1)
        for y in self.unique:
            py = self.yprobs[y]
            pred = self.guess(X,py,XY[XY[:,-1] == y])
            if pred > pre:
                pre = pred
                s_y = y
        return s_y

    

    def argmax(self,preds,xy,cnt):
        op = 0
        for i in range(len(preds)):
            x = preds[i]
            x = np.array(x)
            op = op + log(x.sum())
            
        return op

    def pickProb(self):
        l = []
        pr1 = uniform(0,1)
        pr2 = uniform(0,1 - pr1)
        pr3 = 1 - pr1 - pr2
        l.append(pr1)
        l.append(pr2)
        l.append(pr3)
        return l

    def mixtureNB(self,X,Y,XTrain,YTrain):
        
        ys = Y
        self.unique = np.unique(Y)
        self.X = X
        self.Y = Y
        
        for i in range(10):
            inp = np.append(X, ys.reshape(len(Y), 1), 1)
            prevlt = (-1 * sys.maxint)
            l = None
            count = 0
            start = 0
            while True:
                final_preds = []
                Y_preds = {}
                XYa = []
                if start == 0:
                    start = 1
                    lp = self.pickProb()
                    for iy in range(len(self.unique)):
                        y = self.unique[iy]
                        Y_preds[y] = lp[iy]
                else:
                    for y in self.unique:
                        ylen = len(Y[Y == y])
                        Y_preds[y] = float(ylen)/float(len(Y))
                for x in X:
                    preds = []
                    pred = 0
                    s_y = None
                    for y in self.unique:
                        prediction = self.guess(x,Y_preds[y],inp[inp[:,-1] == y])
                        preds.append(prediction)
                        if prediction > pred:
                            pred = prediction
                            s_y = y
                    final_preds.append(preds)
                    xyn = np.append(x , s_y).tolist()
                    XYa.append(xyn)
                l = self.argmax(final_preds , inp,count)
                inp = np.array(XYa)
                Y = inp[:,-1]
                self.Y = Y
                count = count + 1
                if l - prevlt < 0.0001 or count == 50:
                    for yx in self.unique:
                        predp = float(len(Y[Y == yx])) / float(len(Y))
                        self.yprobs[yx] = predp
                    prevlt = l
                    break
                prevlt = l
            print "Training accuracy"
            print self.acc(X, ys)
            print "Training accuracy"
            print self.acc(XTrain,YTrain)

data = np.genfromtxt("D:\semester_3\machine learning\Assgn\Assgn4\bio.data",delimiter=',',dtype=str)
Xdata = data[:,1:]
Ydata = data[:,0]
Xdata = np.apply_along_axis(split, 1, Xdata)

X = Xdata[:2126,:]
Y = Ydata[:2126]

X_t = Xdata[2126:,:]
Y_t = Ydata[2126:]

naive = MixtureNB()
naive.mixtureNB(X,Y,X_t,Y_t)

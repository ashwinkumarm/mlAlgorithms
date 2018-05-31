# Machine Learning Algorithms

This repository consists of several machine learning algorithms being implemented from scratch using Python and Matlab. Each algorithm is explained breifly below. Datasets are taken from UCI. For each algorithm, UCI dataset name is being mentioned.  

## Primal SVM

 Primal svm is finding the minimum of the following equations. 
                 (1/2) ||w||^2
                 
 We use the following matlab inbuilt quadprog method to solve this quadratic problem. 
 x = quadprog(H,f,A,b);
 
 where 
 x = [w b]
 H = [I 0;0 0]
 A being diagonal(Y) * vector of X alone with last column being vector of ones
 b is vector of 1
 
 ### Dataset
 UCI mystery dataset is being used. 
 
 ### Procedure to run
 File name is primalSVM.m. Just provide the correct path for mystery dataset and run the code. 
 
 Value of weights and bias will be printed along with support vectors. 

# Machine Learning Algorithms

This repository consists of several machine learning algorithms being implemented from scratch using Python and Matlab. Each algorithm is explained breifly below. Most of the datasets are taken from UCI. If data is taken from dataset, then the name is mentioned. Otherwise you can find the dataset in /data folder.

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
 mystery dataset is being used. 
 
 ### Procedure to run & Result shown
 File name is primalSVM.m. Just provide the correct path for mystery dataset and run the code. 
 
 Value of weights and bias will be printed along with support vectors. 
 
 ## Perceptron - Gradient Descent
 
 The below image explains the implementaion details of perceptron. 
 
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/perceptron.png)
 
 ### Dataset
 perceptron dataset is used. 
 
 ### Procedure to run & Result shown
 File name is perceptron/perceptronLearning.py. Just provide the correct path for perceptron dataset and run the code. 
 
 W and B for each iteration is printed. When the value is converged it stops. 
 
 ## Perceptron - Stochastic Gradient Descent
 
 The below image explains the implementaion details of perceptron with stochastic gradient. 
 
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/stochastic_gradient.png)
 
 ### Dataset
 perceptron dataset is used. 
 
 ### Procedure to run & Result shown
 File name is perceptron/stochasticLearning.py. Just provide the correct path for perceptron dataset and run the code. 
 
 W and B are printed once the values are converge. Total number of steps taken to converge are also printed. 
 
 

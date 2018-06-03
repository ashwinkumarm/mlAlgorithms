# Machine Learning Algorithms

This repository consists of several machine learning algorithms being implemented from scratch using Python and Matlab. Each algorithm is explained breifly below. Most of the datasets are taken from UCI. If data is taken from UCI dataset, then the name is mentioned. Otherwise you can find the dataset in /data folder.

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
 
 ## Primal SVM with Slack 
 
 The below image explains the implementaion details of primal svm with slack
 
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/svmwithslack.png)
 
 ### Dataset
 UCI's breast cancer data set is being used. We have randomly separated the dataset to training, validation and test dataset - (0.6, 0.2, 0.2)
 
 ### Procedure to run & Result shown
 File name is primalSvmWithSlack.m. Just provide the correct path for breast cancer dataset and run the code. 
 
 Accuracy for each value of c for training and validation data set will be printed. Model automaticaly fine tunes and would print the result for test data set with tuned hyper parameters. 
 
 
 ## Dual SVM 
 
 The below image explains the implementaion details of Dual svm 
 
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/dual.png)
 
 ### Dataset
 UCI's breast cancer data set is being used. We have randomly separated the dataset to training, validation and test dataset - (0.6, 0.2, 0.2)
 
 ### Procedure to run & Result shown
 File name is dualSvm.m. Just provide the correct path for breast cancer dataset and run the code. 
 
 Accuracy for each value of c for training and validation data set will be printed. Model automaticaly fine tunes and would print the result for test data set with tuned hyper parameters. 
 
 ## K Nearest Neighbours
 
 To find the nearest neighbours, simple distance formula is being used. In future we can K Dimensional Tree to improve the performance of the model.
 
 ### Dataset
 UCI's breast cancer data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is kNearest.py. Just provide the correct path for breast cancer dataset and run the code. 
 
 Accuracy for each value of k for the test data set will be printed.
 
 
 ## Decision Tree -- Information Gain Based
 
 Atrribute split for decision tree is based on the Information Gain for the attributes. Higher the information gain less uncertainity about Y given X. Following image shows the formula for entropy calculation in IG. 
 
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/decisiontree_entropy.png)
 
 ### Dataset
 UCI's Mushroom data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is decisionTreeMushroom.py. Just provide the correct path for mushroom dataset and run the code. 
 
 Accuracy for each training and test data set will be printed.
 
 ## AdaBoost
 
 We have considered hypothesis space to be all possible 3 attribute split decision tree. To increase / decrease the number of iteration, change the value of M accordingly. 
 
 The implemention/algorithm details is explained in the below image. 
  
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/adaboost.png)
 
 ### Dataset
 UCI's Heart data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is adaBoost.py. Just provide the correct path for heart dataset and run the code. 
 
 Accuracy for each round for both training and test data set will be printed.
 

## Naive Bayes
 Our training data is continous, so we have used guassian distribution for conditional probability distribution of each continous feature. 
 
 The implemention/algorithm details is explained in the below image. 
  
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/adaboost.png)
 
 ### Dataset
 UCI's Spambase data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is naive_guassian.m. Just provide the correct path for Spambase dataset and run the code. 
 
 Accuracy for test data set will be printed.
 


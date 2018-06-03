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
  
 ![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/naive_bayes.png)
 
 ### Dataset
 UCI's Spambase data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is naive_guassian.m. Just provide the correct path for Spambase dataset and run the code. 
 
 Accuracy for test data set will be printed.
 
 ## Using PCA for Feature Selection 
 If we performed PCA directly on the training data, we would generate new features that are linear combinations of our original features. If instead, we wanted to find a subset of our current features that were good for classification, we could still use PCA, but we would need to be more clever about it. The primary idea in this approach is to select features from the data that are good at explaining as much of the variance as possible. To do this, we can use the results of PCA as a guide. 

Algorithm for a given k and s is explained by the following image:

![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/pca_featureSelection.png)

### Dataset
 UCI's Spambase data set is being used. We have randomly separated the dataset to training and test dataset.
 
 ### Procedure to run & Result shown
 File name is naivePca.m. Just provide the correct path for Spambase dataset and run the code. 
 
 Accuracy for test data set for eachc K and C will be printed.
 
 ## Constructing a Bayesian Network Model 
 To construct a Bayesian Network model, we need to construct Info Gain Matrix for each edges and pass through a model which finds the maximum spanning tree. We have used matlab inbuit method to construct maximum spanning tree. 
 
Algorithm for constructing Info Gain Matrix is shown below:

![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/infogain_BN.png)

### Dataset
 UCI's congress data set is being used. We have used observation with no missing entry to construct the bayesian network.
 
 ### Procedure to run & Result shown
 File name is bayesianNW.m. Just provide the correct path for congress dataset and run the code. 
 
 The maximum spanning tree will be displayed.
 
 ## Using EM algorithm to learn parameters for a Bayesian Network with missing entries
 We are using the previously constructed bayesian network for this problem, We pass the edges and learn the parameters. First we construct all possible data points if an entry has missing values. Now we find the best point which gives us the maximum probability. We update thethas at the end of a single iteration over entire data set. Den we compare the prev prob of the entire iteration. If it is less than the threshold (converged) we stop. 
 
Algorithm for EM algorithm is shown below:

![alt text](https://github.com/ashwinkumarm/mlAlgorithms/blob/master/images/EM_BN.png)

### Dataset
 UCI's congress data set is being used. We have used observation with missing entries.
 
 ### Procedure to run & Result shown
 File name is bayesianMissingAttr.py. Just provide the correct path for congress dataset and run the code. 
 
 The missing attribute probabilities, log likelihood and learned paramters for each attribute will be printed. 
 
 
 
 ## References: 
 Implementation details are taken from Nicholas Ruozzi lecture slides-- UTD Professor and Tom Mitchell online video and lecture slides -- CMU Professor. 


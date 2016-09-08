## Staff solutions for Project 2 ##

from utils import *
from softmax_SOLN import softmaxRegression, getClassification, plotCostFunctionOverTime, computeTestError
import numpy as np
import matplotlib.pyplot as plt
from features_SOLN import *

# Load MNIST data:
trainX, trainY, testX, testY = getMNISTData()

plotImages(trainX[0:20,:])  # Plot the first 20 images of the training set.

# train softmax, classify test data, compute test error, and plot cost function
def runSoftmaxOnMNIST():
    trainX, trainY, testX, testY = getMNISTData()
    theta, costFunctionHistory = softmaxRegression(trainX, trainY, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
    plotCostFunctionOverTime(costFunctionHistory)
    testError = computeTestError(testX, testY, theta)
    writePickleData(theta, "./theta.pkl.gz")  # Save the model parameters theta obtained from calling softmaxRegression to disk.
    return testError

print 'testError =', runSoftmaxOnMNIST()
# testError should be 0.1005

######################################################
# This section contains the primary code to run when
# working on the "Using manually crafted features" part of the project.
######################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.
n_components = 20
pcs = principalComponents(trainX)
train_pca = projectOntoPC(trainX, pcs, n_components)
test_pca = projectOntoPC(testX, pcs, n_components)
# train_pca (and test_pca) is a representation of our training (and test) data 
# after projecting each example onto the first 20 principal components.

# TODO: Train your softmax regression model using (train_pca, trainY) and evaluate its accuracy on (test_pca, testY).
theta_pca, costFunctionHistory = softmaxRegression(train_pca, trainY, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
plotCostFunctionOverTime(costFunctionHistory) # Do NOT include this plot in write-up.
testError_pca = computeTestError(test_pca, testY, theta_pca)
print 'testError for PCA features =', testError_pca
# testError_pca should be 0.1357

# TODO: Use the plotPC function in features.py to produce scatterplot 
#       of the first 100 MNIST images, as represented in the space spanned by the 
#       first 2 principal components found above.
plotPC(trainX[range(100),], pcs, trainY[range(100)])

# TODO: Use the reconstructPC function in features.py to show
#       the first and second MNIST images as reconstructed solely from 
#       their 20-dimensional principal component representation.
firstimage_reconstructed = reconstructPC(train_pca[0,], pcs, n_components, trainX)
plotImages(firstimage_reconstructed)
plotImages(trainX[0,]) # Compare the reconstructed image with the original.

secondimage_reconstructed = reconstructPC(train_pca[1,], pcs, n_components, trainX)
plotImages(secondimage_reconstructed)
plotImages(trainX[1,]) # Compare the reconstructed image with the original.


## Quadratic Features ##

# TODO: First fill out quadraticFeatures() function in features.py as the below code requires it.
train_quad = quadraticFeatures(train_pca)
test_quad = quadraticFeatures(test_pca)
# train_quad (and test_quad) is a representation of our training (and test) data 
# after applying the quadratic kernel feature mapping to the 20-dimensional PCA representations.


# Note we also accept solutions which removed the column of ones from 
# the quadratic features since this is appended in softmaxRegression:
# train_quad = train_quad[:,:-1]
# test_quad =  test_quad[:,:-1]

# TODO: Train your softmax regression model using (train_quad, trainY) and evaluate its accuracy on (test_quad, testY).
theta_quad, costFunctionHistory = softmaxRegression(train_quad, trainY, alpha= 0.3, lambdaFactor = 1.0e-4, k = 10, numIterations = 150)
plotCostFunctionOverTime(costFunctionHistory) # Do NOT include this plot in write-up.
testError_quad = computeTestError(test_quad, testY, theta_quad)
print 'testError for Quadratic features =', testError_quad
# testError_quad should be 0.0366 (without removing column of ones).
# testError_quad should be 0.0368 (with removing coluumn of ones before calling softmaxRegression.)
 
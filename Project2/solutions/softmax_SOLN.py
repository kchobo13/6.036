from utils import *
import numpy as np
import math
import pdb
import matplotlib.pyplot as plt
import time
import scipy.sparse as sparse

# 0.1015 with column of zeros
# 0.1005 with column of ones
def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta):
    dotProducts = theta.dot(X.T)
    maxOfColumns = dotProducts.max(axis = 0)
    shiftedDotProducts = dotProducts - maxOfColumns
    exponentiated = np.exp(shiftedDotProducts)
    colSums = exponentiated.sum(axis = 0)
    return exponentiated / colSums

def computeCostFunction(X, Y, theta, lambdaFactor):
    numExamples = len(X)
    probabilities = computeProbabilities(X, theta)
    selectedProbabilities = np.choose(Y, probabilities)
    nonRegulizingCost = sum(np.log(selectedProbabilities))
    nonRegulizingCost *= -1.0/numExamples
    regulizingCost = sum(sum(np.square(theta)))
    regulizingCost *= lambdaFactor / 2.0 
    return nonRegulizingCost + regulizingCost

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    numExamples = X.shape[0]
    numLabels = theta.shape[0]
    probabilities = computeProbabilities(X, theta)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*numExamples, (Y,range(numExamples))), shape=(numLabels,numExamples)).toarray()
    nonRegularizedGradient = np.dot(M-probabilities, X)
    nonRegularizedGradient *= -1.0/numExamples 
    return theta - alpha * (nonRegularizedGradient + lambdaFactor * theta)

def softmaxRegression(X, Y, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor)
    return theta, costFunctionProgression

def getClassification(X, theta):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta):
    errorCount = 0.
    assignedLabels = getClassification(X, theta)
    return 1 - np.mean(assignedLabels == Y)


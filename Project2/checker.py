from utils import *
import numpy as np
from softmax_skeleton import augmentFeatureVector, computeProbabilities, computeCostFunction, runGradientDescentIteration

def verifyInputOutputTypesSoftmax():
    trainX, trainY, testX, testY = getMNISTData()
    trainX = augmentFeatureVector(trainX[0:10])
    trainY = trainY[0:10]
    alpha = 0.3
    lambdaFactor = 1.0e-4
    theta = np.zeros([10, trainX.shape[1]])

    # check computeCostFunction
    cost = computeCostFunction(trainX, trainY, theta, lambdaFactor)
    print 'PASSED: computeCostFunction appears to handle correct input type.'
    if isinstance(cost, float):
        print 'PASSED: computeCostFunction appears to return the right type.'
    else:
        print 'FAILED: computeCostFunction appears to return the wrong type. Expected {0} but got {1}'.format(float, type(cost))

    # check computeProbabilities
    probabilities = computeProbabilities(trainX, theta)
    print 'PASSED: computeProbabilities appears to handle correct input type.'
    if isinstance(probabilities, np.ndarray):
        if probabilities.shape == (10, 10):
            print 'PASSED: computeProbabilities return value appears to return a numpy array of the right shape.'
        else:
            print 'FAILED: computeProbabilities return value appears to return a numpy array but with the wrong size. ' + \
                    'Expected a shape of {0} but got {1}.'.format((10,10), probabilities.shape)
    else:
        print 'FAILED: computeProbabilities appears to be the wrong type. ' + \
                'Expected {0} but got {1}.'.format(type(np.array(range(4))), type(probabilities))

    # check gradient descent
    theta = runGradientDescentIteration(trainX, trainY, theta, alpha, lambdaFactor)
    print 'PASSED: runGradientDescentIteration appears to handle correct input type.'
    if isinstance(theta, np.ndarray):
        if theta.shape == (10, 785):
            print 'PASSED: runGradientDescentIteration return value appears to return a numpy array of the right shape.'
        else:
            print 'FAILED: runGradientDescentIteration return value appears to return a numpy array but with the wrong size. ' + \
                    'Expected {0} but got {1}.'.format((10, 785), theta.shape)
    else:
        print 'FAILED: runGradientDescentIteration appears to return the wrong type. ' + \
                'Expected {0} but got {1}'.format(type(np.array(range(4))), type(theta))


verifyInputOutputTypesSoftmax()
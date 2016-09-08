from utils import *
import numpy as np
from softmax_skeleton import augmentFeatureVector, computeProbabilities, computeCostFunction, runGradientDescentIteration

trainX, trainY, testX, testY = getMNISTData()

trainX = augmentFeatureVector(trainX[0:10])
trainY = trainY[0:10]

alpha = 0.3
lambdaFactor = 1.0e-4


def verifyFirstIterationCostFunction():
  theta = np.zeros([10, trainX.shape[1]])
  
  cost = computeCostFunction(trainX, trainY, theta, lambdaFactor)
  if abs(cost-2.30258509299) < 1.0e-6:
    print "Verifying cost during first iteration: Passed"
  else:
    print "Verifying cost during first iteration: Failed"


def verifyFirstIterationProbabilities():
  theta = np.zeros([10, trainX.shape[1]])

  probabilities = computeProbabilities(trainX, theta)
  probabilities_correct = np.ones((10,10), dtype=np.float)*0.1
  if np.all(np.absolute(probabilities-probabilities_correct)< 1.0e-6):
    print "Verifying probabilities during first iteration: Passed"
  else:
    print "Verifying probabilities during first iteration: Failed"


def verifySecondIterationCostFunction():
  theta = np.zeros([10, trainX.shape[1]])

  for i in xrange(2):
    cost = computeCostFunction(trainX, trainY, theta, lambdaFactor)
    theta = runGradientDescentIteration(trainX, trainY, theta, alpha, lambdaFactor)
  
  if abs(cost-0.896839929358) < 1.0e-6:
    print "Verifying cost during second iteration: Passed"
  else:
    print "Verifying cost during second iteration: Failed"


def verifySecondIterationProbabilities():
  theta = np.zeros([10, trainX.shape[1]])

  for i in xrange(2):
    probabilities = computeProbabilities(trainX, theta)
    theta = runGradientDescentIteration(trainX, trainY, theta, alpha, lambdaFactor)

  probabilities_correct = np.array([[0.09597698, 0.4196854, 0.05667909, 0.06235487, 0.06899079, 0.06916521, 0.0349496, 0.06532456, 0.05395287, 0.0534354],
                                    [0.1983818, 0.10531464, 0.06639631, 0.45786242, 0.28722984, 0.17173531, 0.66822544, 0.17628658, 0.49547944, 0.12812545],
                                    [0.06894131, 0.0729084, 0.07108203, 0.07492149, 0.09201906, 0.32024728, 0.04201663, 0.06632575, 0.06420374, 0.07989414],
                                    [0.15999806, 0.11905036, 0.10471412, 0.10113209, 0.09236406, 0.11466912, 0.0524019, 0.40685285, 0.06768243, 0.08783947],
                                    [0.08006051, 0.07574158, 0.48790648, 0.07751524, 0.10680567, 0.13473093, 0.04778263, 0.12621854, 0.06412731, 0.46603252],
                                    [0.29379288, 0.09461314, 0.05543612, 0.06703539, 0.05533667, 0.06447243, 0.04871483, 0.08654557, 0.06685891, 0.0569282],
                                    [0.01879811, 0.01853099, 0.03324019, 0.03119805, 0.02239253,  0.01757959, 0.01861633, 0.01016821, 0.03562178, 0.02229219],
                                    [0.01879811, 0.01853099, 0.03324019, 0.03119805, 0.02239253, 0.01757959, 0.01861633, 0.01016821, 0.03562178, 0.02229219],
                                    [0.01879811, 0.01853099, 0.03324019, 0.03119805, 0.02239253, 0.01757959, 0.01861633, 0.01016821, 0.03562178, 0.02229219],
                                    [0.04645412, 0.0570935, 0.05806529, 0.06558436, 0.23007632, 0.07224095, 0.05006, 0.04194153, 0.08082997, 0.06086825]])
  
  if np.all(np.absolute(probabilities-probabilities_correct) < 1.0e-6):
    print "Verifying probabilities during second iteration: Passed"
  else:
    print "Verifying probabilities during second iteration: Failed"



verifyFirstIterationCostFunction()
verifyFirstIterationProbabilities()
verifySecondIterationCostFunction()
verifySecondIterationProbabilities()


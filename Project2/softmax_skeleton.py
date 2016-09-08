from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta):
    final = []
    for data in X:
        c = np.dot(theta[len(theta)-1], data)
        prob = []
        sum = 0.0
        for thetas in theta:
            probs = math.e**(np.dot(data, thetas)-c)
            sum += probs
            prob.append(probs)

        for x in xrange(len(prob)):
            prob[x] *= 1.0/sum
        final.append(prob)

    final2 = []
    for i in xrange(len(final[0])):
        list1 = []
        for list in final:
            list1.append(list[i])
        final2.append(list1)
    return np.array(final2)

def computeCostFunction(X, Y, theta, lambdaFactor):
    m = len(X)
    t = len(theta)
    x = len(theta[0])
    final = 0.0
    for i in xrange(m):
        sum = 0.0
        for l in xrange(t):
            sum += math.e**(np.dot(X[i], theta[l]))
        for j in xrange(t):
            if Y[i] == j:
                inside = math.e**(np.dot(X[i], theta[j]))/sum
                log = math.log(inside)
                final += log
    final *= -1.0/m

    sum2 = 0.0
    for k in xrange(t):
        for j in xrange(x):
            sum2 += (theta[k][j])**2
    sum2 *= lambdaFactor/2.0

    final += sum2
    return final

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    prob = computeProbabilities(X, theta)
    m = len(X)
    t = len(theta)
    final = []
    for j in xrange(t):
        right = float(lambdaFactor) * theta[j]
        probrow = prob[j]
        left = 0.0
        for i in xrange(m):
            y = 0
            if Y[i] == j:
                y=1
            inside = X[i]*(y - probrow[i])
            left += inside
        total = alpha*-1/m * left + right
        final.append(theta[j] - total)
    return np.array(final)

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
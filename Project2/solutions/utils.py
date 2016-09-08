import cPickle, gzip, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def plotImages(X):
    if X.ndim == 1:
        X = np.array([X])
    numImages = X.shape[0]
    numRows = math.floor(math.sqrt(numImages))
    numCols = math.ceil(numImages/numRows)
    for i in range(numImages):
        reshapedImage = X[i,:].reshape(28,28)
        plt.subplot(numRows, numCols, i+1)
        plt.imshow(reshapedImage, cmap = cm.Greys_r)
        plt.axis('off')
    plt.show()


def pickExamplesOf(X, Y, labels, totalCount):
    boolArr = None
    for label in labels:
        boolArrForLabel = (Y == label)
        if boolArr is None:
            boolArr = boolArrForLabel
        else:
            boolArr |= boolArrForLabel
    filteredX = X[boolArr]
    filteredY = Y[boolArr]
    return (filteredX[:totalCount], filteredY[:totalCount])


def extractTrainingAndTestExamplesWithLabels(trainX, trainY, testX, testY, labels, trainingCount, testCount):
    filteredTrainX, filteredTrainY = pickExamplesOf(trainX, trainY, labels, trainingCount)
    filteredTestX, filteredTestY = pickExamplesOf(testX, testY, labels, testCount)
    return (filteredTrainX, filteredTrainY, filteredTestX, filteredTestY)

def writePickleData(data, fileName):
    f = gzip.open(fileName, 'wb')
    cPickle.dump(data, f)  
    f.close()

def readPickleData(fileName):
    f = gzip.open(fileName, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def getMNISTData():
    trainSet, validSet, testSet = readPickleData('mnist.pkl.gz')   
    trainX, trainY = trainSet
    validX, validY = validSet
    trainX = np.vstack((trainX, validX))
    trainY = np.append(trainY, validY)
    testX, testY = testSet
    return (trainX, trainY, testX, testY)

def loadTrainAndTestPickle(fileName):
    trainX, trainY, testX, testY = readPickleData(fileName)
    return trainX, trainY, testX, testY

# returns the feature set in a numpy ndarray
def loadCSV(filename):
    stuff = np.asarray(np.loadtxt(open(filename, 'rb'), delimiter=','))
    return stuff

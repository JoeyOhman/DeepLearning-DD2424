import numpy as np
import matplotlib.pyplot as plt
import Assignment1.functions as funcs

d = 3072
K = 10


def unpackBatch(fileName):
    batchDict = funcs.loadBatch(fileName)
    data = batchDict[b'data']
    labels = batchDict[b'labels']

    oneHot = []
    for i in range(len(labels)):
        el = np.zeros(10)
        el[labels[i]] = 1
        oneHot.append(el)

    return data.tolist(), oneHot


def unpackData():
    fileNames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    trainX = []
    trainY = []

    for fn in fileNames:
        batchX, batchY = unpackBatch(fn)
        trainX += batchX
        trainY += batchY

    trainX = np.array(trainX).T
    trainY = np.array(trainY).T

    testX, testY = unpackBatch("test_batch")
    testX = np.array(testX).T
    testY = np.array(testY).T

    # Normalize
    meanX = np.expand_dims(np.mean(trainX, axis=1), 1)
    stdX = np.expand_dims(np.std(trainX, axis=1), 1)
    print(meanX.shape, stdX.shape)

    trainX = (trainX - meanX) / stdX
    testX = (testX - meanX) / stdX

    return trainX, trainY, testX, testY


def forward(X, W, b):
    return funcs.softmax((W @ X) + b)


def predict(X, W, b):
    return np.argmax(forward(X, W, b), axis=0)


def evaluateClassifier(X, W, b):
    return forward(X, W, b)


def computeCost(X, Y, W, b, lam):
    numDataPoints = X.shape[1]
    P = evaluateClassifier(X, W, b)
    loss = -np.sum(np.log(np.sum(Y * P, axis=0)))
    pen = lam * np.sum(W * W)
    return (loss / numDataPoints) + pen


def computeAccuracy(X, Y, W, b):
    numDataPoints = X.shape[1]
    y = np.argmax(Y, axis=0)
    predictions = predict(X, W, b)
    numCorrect = np.sum(predictions == y)
    return numCorrect / numDataPoints


def computeGradients(X, Y, P, W, lam):
    numDataPoints = X.shape[1]

    G = -(Y - P)
    dLdW = (1 / numDataPoints) * (G @ X.T)
    # dLdb = (1 / numDataPoints) * (G @ np.ones((numDataPoints, 1)))
    # dLdb = np.expand_dims((1 / numDataPoints) * np.sum(G, axis=1), 1)
    dLdb = np.expand_dims(np.mean(G, axis=1), 1)

    gradW = dLdW + 2 * lam * W
    gradB = dLdb

    return gradW, gradB


def gradDiff(X, Y, P, W, lam, b):
    gradW, gradB = computeGradients(X, Y, P, W, lam)

    numGradW, numGradB = funcs.ComputeGradsNumSlow(X, Y, P, W, b, lam, 1e-6, computeCost)
    diffW = np.abs(gradW - numGradW)
    diffB = np.abs(gradB - numGradB)
    print(np.max(diffW))
    print(np.max(diffB))
    if np.max(diffW) < 1e-6:
        print("Gradient W correct!")
    else:
        print("Gradient W incorrect!")

    if np.max(diffB) < 1e-6:
        print("Gradient b correct!")
    else:
        print("Gradient b incorrect!")


def evaluateModel(xTrain, yTrain, xTest, yTest, W, b, lam):
    lossTrain = computeCost(xTrain, yTrain, W, b, lam)
    lossTest = computeCost(xTest, yTest, W, b, lam)

    print("Training Loss: {}".format(lossTrain))
    print("Test Loss/Accuracy: {}".format(lossTest))
    return lossTrain, lossTest


def miniBatchGD(X, Y, GDParams, W, b, lam, xTest, yTest):
    nBatch, lr, nEpochs = GDParams
    numDataPoints = X.shape[1]
    lossesTrain, lossesTest = [], []

    for epoch in range(nEpochs):

        print("Epoch", epoch + 1)
        lossTrain, lossTest = evaluateModel(X, Y, xTest, yTest, W, b, lam)
        lossesTrain.append(lossTrain)
        lossesTest.append(lossTest)

        # Randomize
        perm = np.random.permutation(numDataPoints)
        X = X[:, perm]
        Y = Y[:, perm]

        # Iterate batches
        batchIndices = np.arange(0, numDataPoints, nBatch)
        for startIdx in batchIndices:
            endIdx = min(startIdx + nBatch, numDataPoints)
            xBatch = X[:, startIdx:endIdx]
            yBatch = Y[:, startIdx:endIdx]
            pBatch = evaluateClassifier(xBatch, W, b)  # forward pass
            gW, gB = computeGradients(xBatch, yBatch, pBatch, W, lam)  # backward pass
            W -= lr * gW
            b -= lr * gB

    # Last update to history
    lossTrain, lossTest = evaluateModel(X, Y, xTest, yTest, W, b, lam)
    lossesTrain.append(lossTrain)
    lossesTest.append(lossTest)

    history = lossesTrain, lossesTest
    return W, b, history


def plotLearningCurves(history):
    labels = ["Training Loss", "Validation Loss"]
    for i, h in enumerate(history):
        plt.plot(h, label=labels[i])

    plt.legend()
    # plt.title("Cost throughout training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def run():
    trainX, trainY, testX, testY = unpackData()
    X = trainX
    Y = trainY

    W = np.random.normal(0, 0.01, (K, d))
    b = np.random.normal(0, 0.01, (K, 1))

    acc = computeAccuracy(testX, testY, W, b)
    print("Accuracy:", acc)
    lam = 0

    # gradDiff(X, Y, P, W, lam, b)
    params = (100, 0.1, 40)
    W, b, history = miniBatchGD(X, Y, params, W, b, lam, testX, testY)

    print("Final Test Accuracy:", computeAccuracy(testX, testY, W, b))
    plotLearningCurves(history)
    funcs.montage(W)


if __name__ == '__main__':
    run()

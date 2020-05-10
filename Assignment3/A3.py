import numpy as np
import matplotlib.pyplot as plt
import Utils.functions as funcs

d = 3072
K = 10
m = 50

N = 49000
BATCH_SIZE = 100

ETA_MIN = 1e-5
ETA_MAX = 1e-1
# N_S = 800

N_S = 2 * np.floor(N / BATCH_SIZE)


def unpackData():
    fileNames = ["data_batch_1"]  # , "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    trainX = []
    trainY = []

    for fn in fileNames:
        batchX, batchY = funcs.unpackBatch(fn)
        trainX += batchX
        trainY += batchY

    trainX = np.array(trainX).T
    trainY = np.array(trainY).T

    valX = trainX[:, -1000:]
    valY = trainY[:, -1000:]
    trainX = trainX[:, :-1000]
    trainY = trainY[:, :-1000]

    testX, testY = funcs.unpackBatch("test_batch")
    testX = np.array(testX).T
    testY = np.array(testY).T

    # Normalize
    meanX = np.expand_dims(np.mean(trainX, axis=1), 1)
    stdX = np.expand_dims(np.std(trainX, axis=1), 1)

    trainX = (trainX - meanX) / stdX
    valX = (valX - meanX) / stdX
    testX = (testX - meanX) / stdX

    return trainX, trainY, valX, valY, testX, testY


def forward(X, W, b):
    H = [X]
    for k in range(len(W) - 1):
        out = np.maximum((W[k] @ H[k]) + b[k], 0)  # ReLu
        H.append(out)

    P = funcs.softmax((W[-1] @ H[-1]) + b[-1])  # Softmax
    H.append(P)
    return H


def predict(X, W, b):
    H = forward(X, W, b)
    return np.argmax(H[-1], axis=0)


def computeCost(X, Y, W, b, lam):
    numDataPoints = X.shape[1]
    H = forward(X, W, b)
    loss = -np.sum(np.log(np.sum(Y * H[-1], axis=0)))
    pen = lam * np.sum([np.sum(wMat * wMat) for wMat in W])
    return (loss / numDataPoints) + pen


def computeAccuracy(X, Y, W, b):
    numDataPoints = X.shape[1]
    y = np.argmax(Y, axis=0)
    predictions = predict(X, W, b)
    numCorrect = np.sum(predictions == y)
    return numCorrect / numDataPoints


def initializeParams(dimensions):
    W = []
    b = []
    prevDim = d
    for i in range(len(dimensions)):
        newW = np.random.normal(0, 1 / np.sqrt(prevDim), (dimensions[i], prevDim))
        W.append(newW)
        b.append(np.zeros((dimensions[i], 1)))

        prevDim = dimensions[i]

    return W, b


def computeGradients(X, Y, W, lam):
    gradW = []
    gradB = []
    numDataPointsInv = 1 / X[0].shape[1]

    G = -(Y - X[-1])

    for k in range(len(W)-1, 1):
        dLdW = numDataPointsInv * (G @ X[k-1].T)
        gW = dLdW + 2 * lam * W[k]
        gB = np.expand_dims(np.mean(G, axis=1), 1)
        gradW.append(gW)
        gradB.append(gB)

        indMat = np.where(X[k-1] > 0, 1, 0)
        G = (W[k].T @ G) * indMat

    dLdW = numDataPointsInv * (G @ X[0].T)
    gW = dLdW + 2 * lam * W[0]
    gB = np.expand_dims(np.mean(G, axis=1), 1)
    gradW.append(gW)
    gradB.append(gB)

    return gradW.reverse(), gradB.reverse()  # TODO: REVERSE! does it work like this? is this what I want?

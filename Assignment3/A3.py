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
    fileNames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

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


# Xavier normalization
def initializeParams(dimensions):
    W = []
    b = []
    for i in range(1, len(dimensions)):
        prevDim = dimensions[i - 1]
        newW = np.random.normal(0, 1 / np.sqrt(prevDim), (dimensions[i], prevDim))
        W.append(newW)
        b.append(np.zeros((dimensions[i], 1)))

    return W, b


def computeGradients(X, Y, W, lam):
    gradW = []
    gradB = []
    numDataPointsInv = 1 / X[0].shape[1]

    G = -(Y - X[-1])

    # print(len(W) - 1)
    for k in range(len(W) - 1, 0, -1):
        dLdW = numDataPointsInv * (G @ X[k].T)
        gW = dLdW + 2 * lam * W[k]
        gB = np.expand_dims(np.mean(G, axis=1), 1)
        gradW.append(gW)
        gradB.append(gB)

        indMat = np.where(X[k] > 0, 1, 0)
        G = (W[k].T @ G) * indMat

    dLdW = numDataPointsInv * (G @ X[0].T)
    gW = dLdW + 2 * lam * W[0]
    gB = np.expand_dims(np.mean(G, axis=1), 1)
    gradW.append(gW)
    gradB.append(gB)

    return gradW[::-1], gradB[::-1]  # Reverse


# TODO: Iterate over k
def gradDiff(X, Y, W, lam, b, numLayers):
    H = forward(X, W, b)
    gradW, gradB = computeGradients(H, Y, W, lam)

    numGradW, numGradB = funcs.computeGradsNum(X, Y, W, b, lam, 1e-5, computeCost, numLayers)

    maxDiffW = 0
    maxDiffB = 0
    for k in range(numLayers):
        diffW = np.max(np.abs(gradW[k] - numGradW[k]))
        diffB = np.max(np.abs(gradB[k] - numGradB[k]))
        maxDiffW = max(diffW, maxDiffW)
        maxDiffB = max(diffB, maxDiffB)

    print(maxDiffW)
    print(maxDiffB)
    if maxDiffW < 1e-6:
        print("Gradient W correct!")
    else:
        print("Gradient W incorrect!")

    if maxDiffB < 1e-6:
        print("Gradient b correct!")
    else:
        print("Gradient b incorrect!")


def plotLearningCurves(history):
    labels = ["Training", "Validation"]
    for k in [0, 2, 4]:
        for i, h in enumerate(history[k:k+2]):
            plt.plot(h, label=labels[i])

        plt.legend()
        plt.xlabel("Epoch")
        plt.show()


def evaluateModel(xTrain, yTrain, xTest, yTest, W, b, lam):
    # These could be optimized by sharing forward props
    costTrain = computeCost(xTrain, yTrain, W, b, lam)
    costTest = computeCost(xTest, yTest, W, b, lam)
    lossTrain = computeCost(xTrain, yTrain, W, b, 0)
    lossTest = computeCost(xTest, yTest, W, b, 0)
    accTrain = computeAccuracy(xTrain, yTrain, W, b)
    accTest = computeAccuracy(xTest, yTest, W, b)

    # print("Training Loss: {}".format(lossTrain))
    # print("Test Loss: {}".format(lossTest))
    return costTrain, costTest, lossTrain, lossTest, accTrain, accTest


def getLearningRate(t):
    diff = ETA_MAX - ETA_MIN
    lCyc = int(t / (2 * N_S))

    if 2 * lCyc * N_S <= t <= (2 * lCyc + 1) * N_S:
        return ETA_MIN + (t - 2 * lCyc * N_S) * diff / N_S
    else:
        return ETA_MAX - (t - (2 * lCyc + 1) * N_S) * diff / N_S


def miniBatchGD(X, Y, GDParams, W, b, lam, xVal, yVal):
    nBatch, lr, nEpochs = GDParams
    numDataPoints = X.shape[1]
    costsTrain, costsTest, lossesTrain, lossesTest, accsTrain, accsTest = [], [], [], [], [], []
    t = 0

    for epoch in range(nEpochs):

        print("Epoch", epoch + 1)
        costTrain, costTest, lossTrain, lossTest, accTrain, accTest = evaluateModel(X, Y, xVal, yVal, W, b, lam)
        costsTrain.append(costTrain)
        costsTest.append(costTest)
        lossesTrain.append(lossTrain)
        lossesTest.append(lossTest)
        accsTrain.append(accTrain)
        accsTest.append(accTest)

        # Randomize
        perm = np.random.permutation(numDataPoints)
        X = X[:, perm]
        Y = Y[:, perm]

        # Iterate batches
        batchIndices = np.arange(0, numDataPoints, nBatch)
        for startIdx in batchIndices:
            lr = getLearningRate(t)
            t += 1
            endIdx = min(startIdx + nBatch, numDataPoints)
            xBatch = X[:, startIdx:endIdx]
            yBatch = Y[:, startIdx:endIdx]
            hBatch = forward(xBatch, W, b)  # forward pass
            gW, gB = computeGradients(hBatch, yBatch, W, lam)  # backward pass
            for k in range(len(W)):
                W[k] -= lr * gW[k]
                b[k] -= lr * gB[k]

    # Last update to history
    costTrain, costTest, lossTrain, lossTest, accTrain, accTest = evaluateModel(X, Y, xVal, yVal, W, b, lam)
    costsTrain.append(costTrain)
    costsTest.append(costTest)
    lossesTrain.append(lossTrain)
    lossesTest.append(lossTest)
    accsTrain.append(accTrain)
    accsTest.append(accTest)

    history = costsTrain, costsTest, lossesTrain, lossesTest, accsTrain, accsTest
    return W, b, history


def train():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    X = trainX[:, :]
    Y = trainY[:, :]

    dims = [d, 50, 30, 20, 20, 10, 10, 10, 10, K]
    W, b = initializeParams(dims)
    numCycles = 5
    numEpochs = int(numCycles * int(N_S * 2 / 450))
    print("NumEpochs:", numEpochs)
    lam = 0.00137
    lr = 0.001

    # gradDiff(X, Y, W, lam, b)
    params = (BATCH_SIZE, lr, numEpochs)
    W, b, history = miniBatchGD(X, Y, params, W, b, lam, valX, valY)

    print("Final Train Accuracy:", computeAccuracy(X, Y, W, b))
    print("Final Test Accuracy:", computeAccuracy(testX, testY, W, b))
    plotLearningCurves(history)


def run():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    # X = trainX[:10, :1]
    X = trainX
    # Y = trainY[:, :1]
    Y = trainY
    dims = [d, m, K]

    W, b = initializeParams(dims)
    # W[0] = W[0][:, :10]
    # print(W[0].shape)
    # print(X.shape)
    lam = 0.01
    # [print(w.shape) for w in W]
    numLayers = len(W)
    gradDiff(X, Y, W, lam, b, numLayers)


if __name__ == '__main__':
    train()

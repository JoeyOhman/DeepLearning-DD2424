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

    ''' 
    trainX, trainY = unpackBatch("data_batch_1")
    trainX = np.array(trainX).T
    trainY = np.array(trainY).T
    '''
    # valX, valY = unpackBatch("data_batch_2")
    # valX = np.array(valX).T
    # valY = np.array(valY).T
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
    H = np.maximum((W[0] @ X) + b[0], 0)  # ReLu
    P = funcs.softmax((W[1] @ H) + b[1])  # Softmax
    return H, P


def predict(X, W, b):
    H, P = forward(X, W, b)
    return np.argmax(P, axis=0)


def computeCost(X, Y, W, b, lam):
    numDataPoints = X.shape[1]
    H, P = forward(X, W, b)
    loss = -np.sum(np.log(np.sum(Y * P, axis=0)))
    pen = lam * (np.sum(W[0] * W[0]) + np.sum(W[1] * W[1]))
    return (loss / numDataPoints) + pen


def computeAccuracy(X, Y, W, b):
    numDataPoints = X.shape[1]
    y = np.argmax(Y, axis=0)
    predictions = predict(X, W, b)
    numCorrect = np.sum(predictions == y)
    return numCorrect / numDataPoints


def initializeParams():
    W1 = np.random.normal(0, 1 / np.sqrt(d), (m, d))
    b1 = np.zeros((m, 1))
    W2 = np.random.normal(0, 1 / np.sqrt(m), (K, m))
    b2 = np.zeros((K, 1))

    return [W1, W2], [b1, b2]


def computeGradients(X, Y, H, P, W, lam):
    numDataPointsInv = 1 / X.shape[1]

    G = -(Y - P)
    dLdW2 = numDataPointsInv * (G @ H.T)
    gradB2 = np.expand_dims(np.mean(G, axis=1), 1)
    gradW2 = dLdW2 + 2 * lam * W[1]

    indMat = np.where(H > 0, 1, 0)
    G = (W[1].T @ G) * indMat

    dLdW1 = numDataPointsInv * (G @ X.T)
    gradB1 = np.expand_dims(np.mean(G, axis=1), 1)
    gradW1 = dLdW1 + 2 * lam * W[0]

    return [gradW1, gradW2], [gradB1, gradB2]


def gradDiff(X, Y, W, lam, b):
    H, P = forward(X, W, b)
    gradW, gradB = computeGradients(X, Y, H, P, W, lam)

    numGradW, numGradB = funcs.computeGradsNum(X, Y, W, b, lam, 1e-5, computeCost, 2)
    diffW = [1, 1]
    diffB = [1, 1]
    diffW[0] = np.max(np.abs(gradW[0] - numGradW[0]))
    diffW[1] = np.max(np.abs(gradW[1] - numGradW[1]))
    diffB[0] = np.max(np.abs(gradB[0] - numGradB[0]))
    diffB[1] = np.max(np.abs(gradB[1] - numGradB[1]))
    maxDiffW = max(diffW[0], diffW[1])
    maxDiffB = max(diffB[0], diffB[1])
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
            hBatch, pBatch = forward(xBatch, W, b)  # forward pass
            gW, gB = computeGradients(xBatch, yBatch, hBatch, pBatch, W, lam)  # backward pass
            W[0] -= lr * gW[0]
            W[1] -= lr * gW[1]
            b[0] -= lr * gB[0]
            b[1] -= lr * gB[1]

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


def trainEvaluate(trainX, trainY, valX, valY, numEpochs, lam):
    W, b = initializeParams()
    params = (BATCH_SIZE, 0.001, numEpochs)
    W, b, history = miniBatchGD(trainX, trainY, params, W, b, lam, valX, valY)
    acc = computeAccuracy(valX, valY, W, b)
    print("Lambda:", lam, ", Validation Accuracy:", acc)


def lambdaSearch():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    numCycles = 3
    numEpochs = numCycles * int(N_S * 2 / 450)

    # lMin = -5
    # lMax = -1
    lMin = np.log10(0.001)
    lMax = np.log10(0.003)
    stepSize = (lMax - lMin) / 7
    lambdas = np.arange(lMin, lMax + stepSize, stepSize)
    lambdas = 10 ** lambdas
    print(lambdas)

    print("Starting grid search...")
    for lam in lambdas:
        trainEvaluate(trainX, trainY, valX, valY, numEpochs, lam)


def train():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    X = trainX[:, :]
    Y = trainY[:, :]

    W, b = initializeParams()
    numCycles = 4.5
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


if __name__ == '__main__':
    # lambdaSearch()
    train()


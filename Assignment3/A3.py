import numpy as np
import matplotlib.pyplot as plt
import Utils.functions as funcs

d = 3072
K = 10
m = 50

N = 45000
BATCH_SIZE = 100

ETA_MIN = 1e-5
ETA_MAX = 1e-1
# N_S = 800

# N_S = 2 * np.floor(N / BATCH_SIZE)
N_S = 5 * np.floor(N / BATCH_SIZE)


def unpackData():
    # fileNames = ["data_batch_1"]  # , "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
    fileNames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

    trainX = []
    trainY = []

    for fn in fileNames:
        batchX, batchY = funcs.unpackBatch(fn)
        trainX += batchX
        trainY += batchY

    trainX = np.array(trainX).T
    trainY = np.array(trainY).T

    valX = trainX[:, -5000:]
    valY = trainY[:, -5000:]
    trainX = trainX[:, :-5000]
    trainY = trainY[:, :-5000]

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


def BN(X, mu=None, var=None):
    n = X.shape[1]
    # mu = np.expand_dims(np.mean(X, axis=1), 1)
    # var = np.expand_dims(np.var(X, axis=1), 1)
    if mu is None:
        mu = np.reshape(np.mean(X, axis=1), (-1, 1))
        var = np.reshape(np.var(X, axis=1) + np.finfo(float).eps, (-1, 1))
        # var = var * (n-1) / n
    '''
    v = []
    m = mu.shape[0]
    for j in range(m):
        sumV = 0
        for i in range(n):
            # print(X.shape)
            # print(j, m)
            sumV += (X[j, i] - mu[j]) ** 2
        v.append(sumV / n)
    v = np.array(v) + np.finfo(float).eps
    # print(v)
    # print(var)
    print(np.mean(np.abs(v - var)))
    # print("Shapes:", mu.shape, var.shape)
    '''

    '''
    normalized = (X - mu) / np.sqrt(var)
    meanS = np.mean(normalized, axis=1)
    varS = np.var(normalized, axis=1)
    print(meanS)
    print(varS)
    print(normalized[:, 0])
    print()
    '''
    return (X - mu) / np.sqrt(var), mu, var


def forward(X, W, b):
    H = [X]
    for k in range(len(W) - 1):
        out = np.maximum((W[k] @ H[k]) + b[k], 0)  # ReLu
        H.append(out)

    P = funcs.softmax((W[-1] @ H[-1]) + b[-1])  # Softmax
    H.append(P)
    return H


def forwardBatch(X, W, b, gamma, beta, muAvgs=None, vsAvgs=None):
    mus = []
    vs = []
    H = [[X]]
    # print("X shape", X.shape)
    # xNorm = normalize(X)
    # H.append(xNorm)
    # print("Mean diff:", np.mean(np.abs(xNorm - X)))
    for k in range(len(W) - 1):
        s = (W[k] @ H[k][-1]) + b[k]
        # print("s shape", s.shape)
        if muAvgs is None:
            sHat, mu, var = BN(s)
        else:
            sHat, mu, var = BN(s, muAvgs[k], vsAvgs[k])

        if muAvgs is None:
            mus.append(mu)
            vs.append(var)

        sTilde = gamma[k] * sHat + beta[k]
        out = np.maximum(sTilde, 0)  # ReLu
        ret = (s, sHat, mu, var, out)
        H.append(ret)

    s = (W[-1] @ H[-1][-1]) + b[-1]
    # sHat, mu, var = BN(s)
    # sTilde = gamma[-1] * sHat + beta[-1]
    out = funcs.softmax(s)
    ret = (s, out)
    H.append(ret)

    mus = np.array(mus)
    vs = np.array(vs)

    return H, mus, vs


def predict(X, W, b, gamma=None, beta=None, mus=None, vs=None):
    batchNorm = gamma is not None
    if batchNorm:
        H, _, _ = forwardBatch(X, W, b, gamma, beta, mus, vs)
        P = H[-1][-1]
    else:
        H = forward(X, W, b)
        P = H[-1]
    return np.argmax(P, axis=0)


def computeCost(X, Y, W, b, lam, gamma=None, beta=None, mus=None, vs=None):
    batchNorm = gamma is not None
    numDataPoints = X.shape[1]
    if batchNorm:
        H, _, _ = forwardBatch(X, W, b, gamma, beta, mus, vs)
        P = H[-1][-1]
    else:
        H = forward(X, W, b)
        P = H[-1]
    loss = -np.sum(np.log(np.sum(Y * P, axis=0)))
    pen = lam * np.sum([np.sum(wMat * wMat) for wMat in W])
    return (loss / numDataPoints) + pen


def computeAccuracy(X, Y, W, b, gamma=None, beta=None, mus=None, vs=None):
    numDataPoints = X.shape[1]
    y = np.argmax(Y, axis=0)
    predictions = predict(X, W, b, gamma, beta, mus, vs)
    numCorrect = np.sum(predictions == y)
    return numCorrect / numDataPoints


# He normalization
def initializeParams(dimensions, he=True):
    W = []
    b = []
    gamma = []
    beta = []
    for i in range(1, len(dimensions)):
        prevDim = dimensions[i - 1]
        if he:
            newW = np.random.normal(0, np.sqrt(2 / prevDim), (dimensions[i], prevDim))
        else:
            sig = 1e-1
            newW = np.random.normal(0, sig, (dimensions[i], prevDim))
        W.append(newW)
        b.append(np.zeros((dimensions[i], 1)))
        if i < len(dimensions) - 1:
            gamma.append(np.ones((dimensions[i], 1)))
            beta.append(np.zeros((dimensions[i], 1)))

    return W, b, gamma, beta


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


def batchNormBackPass(G, s, mu, var, numDataPoints):
    oneRow = np.ones((1, numDataPoints))
    oneCol = np.ones((numDataPoints, 1))
    # sigma1 = (1. / np.power(var, 0.5))
    # sigma2 = (1. / np.power(var, 1.5))
    sigma1 = np.float_power(var, -0.5)
    sigma2 = np.float_power(var, -1.5)
    # print("sigma:", sigma1.shape)
    G1 = G * (sigma1 @ oneRow)
    G2 = G * (sigma2 @ oneRow)
    D = s - (mu @ oneRow)
    c = (G2 * D) @ oneCol
    G = G1 \
        - (1 / numDataPoints) * ((G1 @ oneCol) @ oneRow) \
        - (1 / numDataPoints) * D * (c @ oneRow)
    return G


def computeGradientsBatch(X, Y, W, lam, gamma):
    gradW = []
    gradB = []
    gradGamma = []
    gradBeta = []
    numDataPoints = X[0][0].shape[1]
    numDataPointsInv = 1 / numDataPoints
    # print("Shapes:")
    # for x in X:
    #     print(x[-1].shape)
    # print("***")

    P = X[-1][-1]
    G = -(Y - P)

    dLdW = numDataPointsInv * (G @ X[-2][-1].T)
    # print("Should be adjacent")
    # print("X accessed outside:", len(X) - 1 - 1)  # x accessed here, should be one greater than in first loop
    # print(len(W) - 2 + 1)  # First x accessed in loop
    gW = dLdW + 2 * lam * W[-1]
    gB = np.expand_dims(np.mean(G, axis=1), 1)
    gradW.append(gW)
    gradB.append(gB)

    indMat = np.where(X[-2][-1] > 0, 1, 0)
    G = (W[-1].T @ G) * indMat

    # print(len(W) - 1)
    for k in range(len(W) - 2, -1, -1):
        s, sHat, mu, var, out = X[k + 1]
        # print("X accessed inside:", k+1)
        # s, sHat, mu, var, out = X[k]
        # dJdGamma = numDataPointsInv * (G * X[k][1])
        dJdGamma = np.expand_dims(np.mean((G * sHat), axis=1), 1)
        dJdBeta = np.expand_dims(np.mean(G, axis=1), 1)
        gradGamma.append(dJdGamma)
        gradBeta.append(dJdBeta)
        # *************

        G = G * (gamma[k] @ np.ones((1, numDataPoints)))
        G = batchNormBackPass(G, s, mu, var, numDataPoints)

        # print("prev X:", k)
        prevX = X[k][-1]
        # prevX = X[k-1][-1]
        dLdW = numDataPointsInv * (G @ prevX.T)
        # ********
        gW = dLdW + 2 * lam * W[k]
        gB = np.expand_dims(np.mean(G, axis=1), 1)
        gradW.append(gW)
        gradB.append(gB)

        # if statement?
        if k > 0:
            indMat = np.where(prevX > 0, 1, 0)
            G = (W[k].T @ G) * indMat

    return gradW[::-1], gradB[::-1], gradGamma[::-1], gradBeta[::-1]  # Reverse


def gradDiffBatch(X, Y, W, lam, b, numLayers, gamma, beta):
    H, _, _ = forwardBatch(X, W, b, gamma, beta)
    gradW, gradB, gradGamma, gradBeta = computeGradientsBatch(H, Y, W, lam, gamma)

    numGradW, numGradB, numGradGamma, numGradBeta = funcs.computeGradsNum(X, Y, W, b, lam, 1e-5, computeCost, numLayers, gamma, beta)

    maxDiffW = 0
    maxDiffB = 0
    maxDiffGamma = 0
    maxDiffBeta = 0
    for k in range(numLayers):
        diffW = np.max(np.abs(gradW[k] - numGradW[k]))
        diffB = np.max(np.abs(gradB[k] - numGradB[k]))
        maxDiffW = max(diffW, maxDiffW)
        maxDiffB = max(diffB, maxDiffB)
        print("W, k={}:".format(k), diffW)
        print("b, k={}:".format(k), diffB)

        if k < numLayers-1:
            diffGamma = np.max(np.abs(gradGamma[k] - numGradGamma[k]))
            diffBeta = np.max(np.abs(gradBeta[k] - numGradBeta[k]))
            maxDiffGamma = max(diffGamma, maxDiffGamma)
            maxDiffBeta = max(diffBeta, maxDiffBeta)
            print("Gamma, k={}:".format(k), diffGamma)
            print("Beta, k={}:".format(k), diffBeta)
        print("")

    if maxDiffW < 1e-6:
        print("Gradient W correct!")
    else:
        print("Gradient W incorrect!")

    if maxDiffB < 1e-6:
        print("Gradient b correct!")
    else:
        print("Gradient b incorrect!")

    if maxDiffGamma < 1e-6:
        print("Gradient Gamma correct!")
    else:
        print("Gradient Gamma incorrect!")

    if maxDiffBeta < 1e-6:
        print("Gradient Beta correct!")
    else:
        print("Gradient Beta incorrect!")


def plotLearningCurves(history):
    labels = ["Training", "Validation"]
    for k in [0, 2, 4]:
        for i, h in enumerate(history[k:k + 2]):
            plt.plot(h, label=labels[i])

        plt.legend()
        plt.xlabel("Epoch")
        plt.show()


def evaluateModel(xTrain, yTrain, xTest, yTest, W, b, lam, gamma=None, beta=None, mus=None, vs=None):
    # These could be optimized by sharing forward props
    costTrain = computeCost(xTrain, yTrain, W, b, lam, gamma, beta, mus, vs)
    costTest = computeCost(xTest, yTest, W, b, lam, gamma, beta, mus, vs)
    lossTrain = computeCost(xTrain, yTrain, W, b, 0, gamma, beta, mus, vs)
    lossTest = computeCost(xTest, yTest, W, b, 0, gamma, beta, mus, vs)
    accTrain = computeAccuracy(xTrain, yTrain, W, b, gamma, beta, mus, vs)
    accTest = computeAccuracy(xTest, yTest, W, b, gamma, beta, mus, vs)

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


def miniBatchGD(X, Y, GDParams, W, b, lam, xVal, yVal, gamma=None, beta=None):
    batchNorm = gamma is not None
    # muAvgs = [] if batchNorm else None
    # vsAvgs = [] if batchNorm else None
    muAvgs = None
    vsAvgs = None
    alpha = 0.85
    nBatch, lr, nEpochs = GDParams
    numDataPoints = X.shape[1]
    costsTrain, costsTest, lossesTrain, lossesTest, accsTrain, accsTest = [], [], [], [], [], []
    t = 0

    for epoch in range(nEpochs):

        print("Epoch", epoch + 1)
        costTrain, costTest, lossTrain, lossTest, accTrain, accTest = evaluateModel(
            X, Y, xVal, yVal, W, b, lam, gamma, beta,
            muAvgs,
            vsAvgs)
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
            if batchNorm:
                hBatch, mus, vs = forwardBatch(xBatch, W, b, gamma, beta)  # forward pass
                if muAvgs is None:
                    muAvgs = mus
                    vsAvgs = vs
                else:
                    muAvgs = alpha * muAvgs + (1 - alpha) * mus
                    vsAvgs = alpha * vsAvgs + (1 - alpha) * vs
                gW, gB, gGamma, gBeta = computeGradientsBatch(hBatch, yBatch, W, lam, gamma)  # backward pass
                # print(len(gW), len(gB), len(gGamma), len(gBeta))
                # print(len(gW), len(W))
                for k in range(len(W) - 1):
                    gamma[k] -= lr * gGamma[k]
                    beta[k] -= lr * gBeta[k]
            else:
                hBatch = forward(xBatch, W, b)  # forward pass
                gW, gB = computeGradients(hBatch, yBatch, W, lam)  # backward pass
            for k in range(len(W)):
                W[k] -= lr * gW[k]
                b[k] -= lr * gB[k]

    # Last update to history
    costTrain, costTest, lossTrain, lossTest, accTrain, accTest = evaluateModel(X, Y, xVal, yVal, W, b, lam, gamma, beta, muAvgs, vsAvgs)
    costsTrain.append(costTrain)
    costsTest.append(costTest)
    lossesTrain.append(lossTrain)
    lossesTest.append(lossTest)
    accsTrain.append(accTrain)
    accsTest.append(accTest)

    history = costsTrain, costsTest, lossesTrain, lossesTest, accsTrain, accsTest
    return W, b, gamma, beta, history, muAvgs, vsAvgs


def train():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    X = trainX[:, :]
    Y = trainY[:, :]

    # dims = [d, 50, 30, 20, 20, 10, 10, 10, 10, K]  # 0.5297
    dims = [d, m, m, K]  # 0.5418
    W, b, gamma, beta = initializeParams(dims, False)
    numCycles = 2
    numEpochs = int(numCycles * int(N_S * 2 / 450))
    print("NumEpochs:", numEpochs)
    # lam = 0.00137
    lam = 0.0075
    lr = 0.001
    # gamma, beta = None, None

    params = (BATCH_SIZE, lr, numEpochs)
    W, b, gamma, beta, history, muAvgs, vsAvgs = miniBatchGD(X, Y, params, W, b, lam, valX, valY, gamma, beta)

    print("Final Train Accuracy:", computeAccuracy(X, Y, W, b, gamma, beta, muAvgs, vsAvgs))
    print("Final Test Accuracy:", computeAccuracy(testX, testY, W, b, gamma, beta, muAvgs, vsAvgs))
    plotLearningCurves(history)


def checkGradients():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    X = trainX[:7, :100]
    # X = trainX
    Y = trainY[:, :100]
    # Y = trainY
    # dims = [d, m, K]
    dims = [d, 13, 11, K]
    # dims = [7, 14, 10, K]
    # dims = [7, m, m, 25, 20, 15, K]

    W, b, gamma, beta = initializeParams(dims)
    W[0] = W[0][:, :7]
    # print(W[0].shape)
    # print(X.shape)
    # lam = 0.01
    lam = 0.005
    # [print(w.shape) for w in W]
    numLayers = len(W)
    gradDiffBatch(X, Y, W, lam, b, numLayers, gamma, beta)


def trainEvaluate(trainX, trainY, valX, valY, numEpochs, lam):
    dims = [d, m, m, K]
    W, b, gamma, beta = initializeParams(dims)
    params = (BATCH_SIZE, 0.001, numEpochs)
    W, b, gamma, beta, history, muAvgs, vsAvgs = miniBatchGD(trainX, trainY, params, W, b, lam, valX, valY, gamma, beta)
    acc = computeAccuracy(valX, valY, W, b, gamma, beta, muAvgs, vsAvgs)
    print("Lambda:", lam, ", Validation Accuracy:", acc)


def lambdaSearch():
    trainX, trainY, valX, valY, testX, testY = unpackData()
    numCycles = 3
    numEpochs = numCycles * int(N_S * 2 / 450)

    # lMin = -5
    # lMax = -1
    lMin = 0.0005
    lMax = 0.025
    stepSize = (lMax - lMin) / 7
    lambdas = np.arange(lMin, lMax + stepSize, stepSize)
    lambdas = lambdas
    print(lambdas)

    print("Starting grid search...")
    for lam in lambdas:
        trainEvaluate(trainX, trainY, valX, valY, numEpochs, lam)


if __name__ == '__main__':
    # checkGradients()
    train()
    # lambdaSearch()

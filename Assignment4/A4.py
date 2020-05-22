import numpy as np
import pickle
import matplotlib.pyplot as plt

from Assignment4.Classes import Vocabulary, RNN, idx2OneHot

epochs = 5
m = 250  # Hidden state size
eta = 0.1  # Learning rate
seqLen = 25  # training sequence size
eps = 1e-10

rnnFileName = "trainedRNN.obj"


def saveRNN(rnn):
    fileHandler = open(rnnFileName, 'wb')
    pickle.dump(rnn, fileHandler)


def loadRNN():
    fileHandler = open(rnnFileName, 'rb')
    return pickle.load(fileHandler)


def readParseTextFile():
    with open('../Datasets/goblet_book.txt', 'r') as file:
        data = file.read().replace('\n', '')
    textArray = np.array(list(data))
    vocab = Vocabulary(textArray)
    idxArray = np.array([vocab.char2idx[c] for c in textArray])
    return vocab, textArray, idxArray


def synthesizeText(vocab, rnn, h0=None):
    genLen = 1000
    x0 = idx2OneHot([vocab.char2idx["."]], vocab.size)
    idxs = rnn.synthesize(genLen, x0, h0).flatten()
    chars = [vocab.idx2char[idx] for idx in idxs]
    text = ""
    for c in chars:
        text += c
        if c == '.':
            text += "\n"
    return text


def train(vocab, rnn, data):
    # data = data[:6000]
    numDataPoints = len(data)
    print(numDataPoints)

    hPrev = None
    smoothLoss = None
    mV, mW, mU, mB, mC = 0, 0, 0, 0, 0

    updates = 0
    smoothLosses, updateIdxs = [], []
    for epoch in range(1, epochs+1):
        for e in range(0, numDataPoints - seqLen, seqLen):
            updates += 1
            lastIdx = min(e + seqLen + 1, numDataPoints - seqLen - 1)
            x = data[e:lastIdx-1]
            y = data[e + 1:lastIdx]
            X = idx2OneHot(x, vocab.size)
            Y = idx2OneHot(y, vocab.size)
            fw, bp, loss = rnn.forwardBackwardPass(len(x), X, Y, hPrev)
            hPrev = fw.h[:, -1].reshape(-1, 1)
            if smoothLoss is None:
                smoothLoss = loss
            else:
                smoothLoss = 0.999 * smoothLoss + 0.001 * loss

            if updates % 2000:
                smoothLosses.append(smoothLoss)
                updateIdxs.append(updates)
            if updates % 5000 == 0:
                print("epoch: {}, updates: {}, smoothLoss: {}, percentageOfEpochDone: {}%"
                      .format(epoch, updates, smoothLoss, 100 * e / numDataPoints))

            # AdaGrad
            mV = mV + np.square(bp.dLdV)
            mW = mW + np.square(bp.dLdW)
            mU = mU + np.square(bp.dLdU)
            mB = mB + np.square(bp.dLdB)
            mC = mC + np.square(bp.dLdC)
            rnn.V = rnn.V - (eta / np.sqrt(mV + eps)) * bp.dLdV
            rnn.W = rnn.W - (eta / np.sqrt(mW + eps)) * bp.dLdW
            rnn.U = rnn.U - (eta / np.sqrt(mU + eps)) * bp.dLdU
            rnn.b = rnn.b - (eta / np.sqrt(mB + eps)) * bp.dLdB
            # print(rnn.c.shape, bp.d)
            rnn.c = rnn.c - (eta / np.sqrt(mC + eps)) * bp.dLdC

            if updates % 10000 == 0:
                print("\n", synthesizeText(vocab, rnn), "\n")
            if updates == 100000:
                break

    saveRNN(rnn)

    return rnn, [smoothLosses, updateIdxs]


def maxRelError(gn, ga, paramName):
    absErr = np.sum(np.abs(ga - gn))
    absErrSum = np.sum(np.abs(ga)) + np.sum(np.abs(gn))
    relErr = absErr / max(1e-4, absErrSum)
    print(str(paramName) + ":", relErr)


def gradDiff(rnn, data, vocab):
    x = data[0:seqLen]
    y = data[1:seqLen + 1]
    X = idx2OneHot(x, vocab.size)
    Y = idx2OneHot(y, vocab.size)

    _, bp, _ = rnn.forwardBackwardPass(seqLen, X, Y)
    numGrads = numericalGradients(X, Y, rnn)
    num_dV, num_dW, num_dU, num_dB, num_dC = numGrads
    dV = bp.dLdV
    dW = bp.dLdW
    dU = bp.dLdU
    dB = bp.dLdB
    dC = bp.dLdC

    def printMaxDiff(x, y, paramName): print(paramName, np.max(np.abs(x - y)))

    printMaxDiff(num_dV, dV, "V")
    printMaxDiff(num_dW, dW, "W")
    printMaxDiff(num_dU, dU, "U")
    printMaxDiff(num_dB, dB, "B")
    printMaxDiff(num_dC, dC, "C")
    maxRelError(num_dV, dV, "V")
    maxRelError(num_dW, dW, "W")
    maxRelError(num_dU, dU, "U")
    maxRelError(num_dB, dB, "B")
    maxRelError(num_dC, dC, "C")


def numericalGradients(X, Y, rnn, h=1e-4):
    numV = np.zeros(rnn.V.shape)
    for i in range(rnn.V.shape[0]):
        for j in range(rnn.V.shape[1]):
            orgVal = rnn.V[i, j]

            rnn.V[i, j] = orgVal - h
            _, l1 = rnn.forwardPass(seqLen, X, Y)
            rnn.V[i, j] = orgVal + h
            _, l2 = rnn.forwardPass(seqLen, X, Y)
            numV[i, j] = (l2 - l1) / (2 * h)

            rnn.V[i, j] = orgVal

    # W
    numW = np.zeros(rnn.W.shape)
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            orgVal = rnn.W[i, j]

            rnn.W[i, j] = orgVal - h
            _, l1 = rnn.forwardPass(seqLen, X, Y)
            rnn.W[i, j] = orgVal + h
            _, l2 = rnn.forwardPass(seqLen, X, Y)
            numW[i, j] = (l2 - l1) / (2 * h)

            rnn.W[i, j] = orgVal

    # U
    numU = np.zeros(rnn.U.shape)
    for i in range(rnn.U.shape[0]):
        for j in range(rnn.U.shape[1]):
            orgVal = rnn.U[i, j]

            rnn.U[i, j] = orgVal - h
            _, l1 = rnn.forwardPass(seqLen, X, Y)
            rnn.U[i, j] = orgVal + h
            _, l2 = rnn.forwardPass(seqLen, X, Y)
            numU[i, j] = (l2 - l1) / (2 * h)

            rnn.U[i, j] = orgVal

    # b
    numB = np.zeros(rnn.b.shape)
    for i in range(numB.shape[0]):
        for j in range(numB.shape[1]):
            orgVal = rnn.b[i, j]

            rnn.b[i, j] = orgVal - h
            _, l1 = rnn.forwardPass(seqLen, X, Y)
            rnn.b[i, j] = orgVal + h
            _, l2 = rnn.forwardPass(seqLen, X, Y)
            numB[i, j] = (l2 - l1) / (2 * h)

            rnn.b[i, j] = orgVal

    # c
    numC = np.zeros(rnn.c.shape)
    for i in range(numC.shape[0]):
        for j in range(numC.shape[1]):
            orgVal = rnn.c[i, j]

            rnn.c[i, j] = orgVal - h
            _, l1 = rnn.forwardPass(seqLen, X, Y)
            rnn.c[i, j] = orgVal + h
            _, l2 = rnn.forwardPass(seqLen, X, Y)
            numC[i, j] = (l2 - l1) / (2 * h)

            rnn.c[i, j] = orgVal

    # print("Arrays are equal V:", np.array_equal(VBackup, rnn.V))
    # print("Arrays are equal W:", np.array_equal(WBackup, rnn.W))
    return numV, numW, numU, numB, numC


def plotLearningCurves(history):
    l, u = history
    plt.plot(u, l)

    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.show()


def run():
    vocab, textArr, idxArr = readParseTextFile()

    rnn = RNN(m, vocab.size, seqLen)
    # rnn = loadRNN()

    # gradDiff(rnn, idxArr, vocab)
    rnn, smoothLosses = train(vocab, rnn, idxArr)

    text = synthesizeText(vocab, rnn)
    print(text)
    plotLearningCurves(smoothLosses)


if __name__ == '__main__':
    run()

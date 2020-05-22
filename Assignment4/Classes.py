import numpy as np

import Utils.functions as funcs

SIG_INIT = 0.01


# Returns a matrix with one-hot columns
def idx2OneHot(idxs, size):
    oneHots = []
    for idx in idxs:
        oneHot = np.zeros(size)
        oneHot[idx] = 1
        oneHots.append(oneHot)
    return np.array(oneHots).T


class Vocabulary:
    def __init__(self, data):
        self.size = 0
        self.char2idx = {}
        self.idx2char = {}
        self.createDict(data)

    def createDict(self, data):
        for char in data:
            if char not in self.char2idx:
                self.char2idx[char] = self.size
                self.idx2char[self.size] = char
                self.size += 1


class Forward:
    def __init__(self, W, U, V, b, c, seqSize, vocabSize, xObs=None, x0=None, h0=None):
        x = x0  # if not synthesizing, this will be overridden in first iteration
        if h0 is None:
            h0 = np.zeros(b.shape)
        h = h0
        aSeq = []
        hSeq = []
        oSeq = []
        pSeq = []
        xIdxSeq = []
        hSeq.append(h0)
        for t in range(seqSize):
            if xObs is not None:
                x = xObs[:, [t]]
            x = x.reshape((-1, 1))
            a = W @ h + U @ x + b
            aSeq.append(a)
            h = np.tanh(a)
            hSeq.append(h)
            o = V @ h + c
            oSeq.append(o)
            p = funcs.softmax(o)
            pSeq.append(p)
            if xObs is None:
                pBetter = p * np.sqrt(p)
                pBetter = pBetter / np.sum(pBetter)
                xIdx = np.random.choice(vocabSize, 1, p=pBetter.T[0])
                xIdxSeq.append(xIdx)
                x = idx2OneHot(xIdx, vocabSize)

        self.seqSize = seqSize
        self.a = np.squeeze(np.array(aSeq)).T
        self.h = np.squeeze(np.array(hSeq)).T
        self.o = np.squeeze(np.array(oSeq)).T
        self.p = np.squeeze(np.array(pSeq)).T
        self.xIdx = np.squeeze(np.array(xIdxSeq))


class BackPropagation:
    def __init__(self, fw, X, Y, tau, rnn):
        m = rnn.m
        K = rnn.vocabSize
        Hwith0 = fw.h[:, :-1]
        HwithLast = fw.h[:, 1:]

        G = - (Y - fw.p).T  # dLdO, tau rows (one row per time step)
        self.dLdV = G.T @ HwithLast.T
        self.dLdC = np.sum(G, axis=0, keepdims=True).T

        # H & A
        dLdHtau = G[-1] @ rnn.V
        dLdH = [dLdHtau]
        dLdAtau = dLdHtau * (1 - np.square(np.tanh(fw.a[:, -1])))
        dLdA = [dLdAtau]
        for t in range(tau - 2, -1, -1):
            dLdht = G[t] @ rnn.V + dLdA[-1] @ rnn.W
            dLdat = dLdht * (1 - np.square(np.tanh(fw.a[:, t])))
            dLdH.append(dLdht)
            dLdA.append(dLdat)

        dLdA = np.array(dLdA[::-1])

        self.dLdB = np.sum(dLdA, axis=0, keepdims=True).T

        G = dLdA

        self.dLdW = G.T @ Hwith0.T

        self.dLdU = G.T @ X.T


class RNN:
    # d = vocabSize
    # K = vocabSize
    # x has shape (vocabSize, seqSize)
    def __init__(self, m, vocabSize, seqSize):
        self.m = m
        self.vocabSize = vocabSize
        self.seqSize = seqSize
        self.b = np.zeros((m, 1))
        self.c = np.zeros((vocabSize, 1))
        self.U = np.random.normal(0, 1, (m, vocabSize)) * SIG_INIT
        self.W = np.random.normal(0, 1, (m, m)) * SIG_INIT
        self.V = np.random.normal(0, 1, (vocabSize, m)) * SIG_INIT

    def synthesize(self, generationLength, x0, h0=None):
        forward = Forward(self.W, self.U, self.V, self.b, self.c, generationLength, self.vocabSize,
                          xObs=None, x0=x0, h0=h0)
        return forward.xIdx

    def forwardPass(self, seqLength, xObs, Y, h0=None):
        fw = Forward(self.W, self.U, self.V, self.b, self.c, seqLength, self.vocabSize, xObs, h0=h0)
        loss = -np.sum(np.log(np.sum(Y * fw.p, axis=0)))
        return fw, loss

    def forwardBackwardPass(self, seqLength, X, Y, h0=None):
        fw, loss = self.forwardPass(seqLength, X, Y, h0)
        bp = BackPropagation(fw, X, Y, seqLength, self)
        return fw, bp, loss

import numpy as np


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def loadBatch(filename):
    """ Copied from the dataset website """
    import pickle

    with open('../Datasets/cifar-10-batches-py/' + filename, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def unpackBatch(fileName):
    batchDict = loadBatch(fileName)
    data = batchDict[b'data']
    labels = batchDict[b'labels']

    oneHot = []
    for i in range(len(labels)):
        el = np.zeros(10)
        el[labels[i]] = 1
        oneHot.append(el)

    return data.tolist(), oneHot


def computeGradsNum(X, Y, W, b, lam, h, cost, K):
    dW = [np.zeros(w.shape) for w in W]
    db = [np.zeros(bi.shape) for bi in b]

    for k in range(K):
        for i in range(b[k].shape[0]):
            b_try = [bi.copy() for bi in b]
            b_try[k][i] -= h
            c1 = cost(X, Y, W, b_try, lam)

            b_try = [bi.copy() for bi in b]
            b_try[k][i] += h
            c2 = cost(X, Y, W, b_try, lam)
            db[k][i] = (c2 - c1) / (2 * h)

            for j in range(W[k].shape[1]):
                W_try = [w.copy() for w in W]
                W_try[k][i, j] -= h
                c1 = cost(X, Y, W_try, b, lam)

                W_try = [w.copy() for w in W]
                W_try[k][i, j] += h
                c2 = cost(X, Y, W_try, b, lam)
                dW[k][i, j] = (c2 - c1) / (2 * h)

    return dW, db


def ComputeGradsNum(X, Y, W, b, lamda, h, computeCost, K):
    """ Converted from matlab code """
    # no = W.shape[0]
    # d = X.shape[0]

    gWList, gBList = [], []

    for k in range(K):
        grad_W = np.zeros(W[k].shape)
        grad_b = np.zeros(b[k].shape)

        c = computeCost(X, Y, W, b, lamda)

        originalB = b[k]
        copyB = np.copy(b[k])
        b[k] = copyB

        for i in range(len(b[k])):
            # b_try = np.array(b[k])
            b[k][i] += h
            c2 = computeCost(X, Y, W, b, lamda)
            grad_b[i] = (c2 - c) / h

        b[k] = originalB

        originalW = W[k]
        copyW = np.copy(W[k])
        W[k] = copyW

        for i in range(W[k].shape[0]):
            for j in range(W[k].shape[1]):
                # W_try = np.array(W)
                W[k][i, j] += h
                c2 = computeCost(X, Y, W, b, lamda)
                grad_W[i, j] = (c2 - c) / h

        W[k] = originalW

        gWList.append(grad_W)
        gBList.append(grad_b)

    return [gWList, gBList]


def ComputeGradsNumSlow(X, Y, W, b, lamda, h, computeCost):
    """ Converted from matlab code """
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = computeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = computeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = computeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = computeCost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


def montage(W):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            # ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()


# def save_as_mat(data, name="model"):
#    """ Used to transfer a python model to matlab """
#    import scipy.io as sio
#    sio.savemat(name'.mat', {name: b})

'''
    dW = [np.zeros(w.shape) for w in W]
    db = [np.zeros(bi.shape) for bi in b]

    for k in [0, 1]:
        for i in range(b[k].shape[0]):
            C = []
            for m in [-1, 1]:
                b_try = [bi.copy() for bi in b]
                b_try[k][i] += m * h
                C.append(computeCost(X, Y, W, b_try, lam))
            db[k][i] = (C[-1] - C[0]) / (2 * h)

            for j in range(W[k].shape[1]):
                C = []
                for m in [-1, 1]:
                    W_try = [w.copy() for w in W]
                    W_try[k][i, j] += m * h
                    C.append(computeCost(X, Y, W_try, b, lam))
                dW[k][i, j] = (C[-1] - C[0]) / (2 * h)
    return dW, db
    '''

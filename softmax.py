from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入具有维度 D，共有 C 个类别，且我们基于包含 N 个样本的小批量（minibatch）进行操作

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        # 当前样本对所有类别的分数，形状为 (C,)
        scores = X[i].dot(W)
        # 数值稳定性处理
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores)
        probabilities = exp_scores / sum_exp_scores
        # 当前样本的交叉熵损失
        loss += -scores[y[i]] + np.log(sum_exp_scores)
        # 计算每一个类别权重的梯度
        for j in range(num_classes):
            coefficient = probabilities[j]

            if j == y[i]:
                coefficient -= 1

            dW[:, j] += coefficient * X[i]
    # 对所有训练样本取平均
    loss /= num_train
    dW /= num_train
    # L2 正则化
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]

    # 1. 一次性计算所有样本的分类分数
    scores = X.dot(W)

    # 2. 每一行减去该行最大值
    scores -= np.max(scores, axis=1, keepdims=True)

    # 3. 计算所有类别的 Softmax 概率
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp_scores

    # 4. 计算平均数据损失
    correct_class_scores = scores[np.arange(num_train), y]
    losses = -correct_class_scores + np.log(sum_exp_scores[:, 0])

    loss = np.mean(losses)
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # 5. 计算分数梯度
    dscores = probabilities.copy()
    dscores[np.arange(num_train), y] -= 1
    dscores /= num_train

    # 6. 根据链式法则计算权重梯度
    dW = X.T.dot(dscores)
    dW += 2 * reg * W

    return loss, dW

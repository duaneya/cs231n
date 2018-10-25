import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)
        loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) / np.sum(np.exp(scores))) * X[i]
            if j == y[i]:
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    loss = -scores[np.arange(X.shape[0]), y] + np.log(np.sum(np.exp(scores), axis=1))
    loss = np.mean(loss)

    prob = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    prob[np.arange(num_train), y] += -1
    dW = X.T.dot(prob)
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    # num_classes = W.shape[1]
    # num_train = X.shape[0]
    # scores = X.dot(W)
    # softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    # loss = -np.sum(np.log(softmax_output[np.arange(num_train), y]))
    # loss /= num_train
    # loss += reg * np.sum(W * W)
    # print(softmax_output.shape)
    # softmax_output[np.arange(num_train), y] += -1
    # dW = (X.T).dot(softmax_output)
    # dW = dW / num_train + 2 * reg * W

    return loss, dW

import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1] # number of classes
  num_train = X.shape[0] # num of training examples
  loss = 0.0 # initializing loss as zero
  # iterating over training examples
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] # correct class score
    count = 0 # initializing count to zero
    # iterating over classes
    for j in range(num_classes):
      # In SVM loss function it will not iterate if both are correct class 
      if j == y[i]:
        continue
      # margins
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # performing max operation 
      if margin > 0:
        # increment the count 
        count += 1
        # incrementing loss
        loss += margin
        #  incrementing gradient
        dW[:,j] += X[i]

    # Correct Class
    dW[:,y[i]] += -count * X[i] 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # averaging gradient
  dW /= num_train
  # adding regularization 
  dW += reg * W 

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # scores
  scores = X.dot(W)
  # correct class score 
  correct_class_score = scores[np.arange(X.shape[0]),y]
  # margin
  margin = np.maximum(0,scores - correct_class_score[:,np.newaxis] + 1)
  # correcting correct class scores as 0 from 1 
  margin[np.arange(X.shape[0]),y] = 0
  # updating loss 
  loss = np.sum(margin)
  # averaging loss by number of training examples
  loss /= X.shape[0]
  # adding regularization 
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  X_mask = np.zeros(margin.shape) # creating a mask with shape of margin
  # For all the values greater than zero initializing to 1. 
  X_mask[margin>0] = 1 
  # Update the count.
  count = np.sum(X_mask,axis = 1)
  # Adding  -(count) for correct class  
  X_mask[np.arange(X.shape[0]),y] -=  count 
  # Now update the gradient 
  dW = X.T.dot(X_mask)
  # averaging with number of training samples
  dW /= X.shape[0]
  # Adding regularization 
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

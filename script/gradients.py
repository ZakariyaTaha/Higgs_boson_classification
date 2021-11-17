import numpy as np
from helpers import sigmoid

def compute_gradient(y, tx, w):
  '''Compute the gradient of the loss
  
  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector

  Returns:
  float: gradient of the loss
  
  '''
  e = y - tx.dot(w)
  return -tx.T.dot(e)/tx.shape[0]


def compute_logistic_gradient(y, tx, w):
  '''Compute the gradient of the loss for logistic regression
  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector

  Returns:
  float: gradient of the loss for logistic regression
  '''

  p = sigmoid(tx.dot(w))
  return tx.T.dot(p - y)


def compute_reg_logistic_gradient(y, tx, w, lambda_):
  '''Compute the gradient of the loss for regularized logistic regression
  
  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector

  Returns:
  float: gradient of the loss for regularized logistic regression
  
  '''
  return compute_logistic_gradient(y, tx, w) + lambda_*w  




import numpy as np
from helpers import sigmoid


def compute_loss(y, tx, w):
  '''Compute the loss using MSE

  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector

  Returns:
  float: MSE loss
  '''
  e = y - tx.dot(w)
  return np.mean(e**2)/2


def compute_logistic_loss(y, tx, w):

  '''Compute the loss for the logistic regression
  
  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector

  Returns:
  float: logistic loss
  '''
  y2 = (y+1)/2
  p = sigmoid(tx.dot(w))
  return - y2.T.dot(np.log(p)) - (1-y2).T.dot(np.log(1-p))


def compute_reg_logistic_loss(y, tx, w, lambda_):
  '''Compute the loss for the regularized logistic regression
  
  Parameters:

  y  : output data
  tx : input data 
  w : weigths  vector
  lambda_ : penalty term

  Returns:
  float: regularized logistic loss
  
  '''
  return compute_logistic_loss(y, tx, w) + lambda_*np.linalg.norm(w, 2)/2





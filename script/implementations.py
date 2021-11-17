import numpy as np
from costs import *
from gradients import *

def least_squares_GD(y, tx, initial_w=None, max_iters=200, gamma=1e-10):
    """
    Linear regression using gradient descent
  
    Parameters:

    y  : output data
    tx : input data 
    initial_w : weigths  vector
    max_iters : number of iteration
    gamma : step-size parameter for gradient descent algorithm

    Returns:
    The optimal weights using gradient descent algorithm and the associated loss
  
    """

    
    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])

    assert max_iters >= 0, "max_iters must be >= 0"

    assert tx.shape[1] == len(initial_w), (
        "len of initial_w ({initial_w_len}) and columns of tx"
        "({tx_cols_len}) don't match".format(
            initial_w_len=len(initial_w),
            tx_cols_len=tx.shape[1]
        )
    )
        

    w = initial_w

    for _ in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad
        
    loss = compute_loss(y, tx, w)
    
    return w, loss


def least_squares_SGD(y, tx, initial_w=None, max_iters=200, gamma=0.1):
    """
    Linear regression using stochastic gradient descent
  
    Parameters:

    y  : output data
    tx : input data 
    initial_w : weigths  vector
    max_iters : number of iteration
    gamma : step-size parameter for gradient descent algorithm

    Returns:
    The optimal weights using stochatic gradient descent algorithm and the associated loss
  
    """

    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])

    N = len(y)
    w = initial_w

    for _ in range(max_iters):
        id = randint(0, N-1)
        w = w - gamma * compute_gradient(y[id], tx[id], w)
    
    loss = compute_loss(y, tx, w)

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations
  
    Parameters:

    y  : output data
    tx : input data 

    Returns:
    The least squares weights and the associated loss
  
    """

    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_=1e-4):
    """
    Ridge regression using normal equations
  
    Parameters:

    y  : output data
    tx : input data 
    lambda_ : penalization parameter

    Returns:
    The optimal weights using ridge regression and the associated loss (using normal equations)

    """

    N, M = tx.shape

    a = tx.T.dot(tx) + 2*N*lambda_ * np.identity(M)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w=None, max_iters=200, gamma=1e-10):
    """
    Logistic regression using gradient descent or SGD
  
    Parameters:

    y  : output data
    tx : input data 
    initial_w : weigths  vector
    max_iters : number of iteration
    gamma : step-size parameter for gradient descent algorithm

    Returns:
    The optimal weights using gradient descent with logistic loss function and the associated loss
  
    """

    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])
    
    w = initial_w
    
    for _ in range(max_iters):

        grad = gamma*compute_logistic_gradient(y, tx, w)
        w = w - grad
        
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w=None, max_iters=200, gamma=1e-10):
    """
    Regularized logistic regression using GD or SGD

    Parameters:

    y  : output data
    tx : input data 
    initial_w : weigths  vector
    max_iters : number of iteration
    gamma : step-size parameter for gradient descent algorithm

    Returns:
    The optimal weights using stochatic gradient descent with logistic loss function and the associated loss
  
    """


    if initial_w is None:
        initial_w = np.zeros(tx.shape[1])

    w = initial_w
    loss = compute_reg_logistic_loss(y, tx, initial_w, lambda_)

    for _ in range(max_iters):

        w = w - gamma*compute_reg_logistic_gradient(y, tx, w, lambda_)
        
        
    loss = compute_reg_logistic_loss(y, tx, initial_w, lambda_)

    return w, loss





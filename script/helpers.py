import numpy as np

def build_poly(x, degree):
    """
    Polynomial basis functions for input data
  
    Parameters:

    tx : input data 
    degree : polynomial degree

    Returns:
    Polynomial basis of the underlying degree
  
    """

    N = x.shape[0]
    poly = np.ones((N, 1))
    for deg in range(degree):
        poly = np.c_[poly, np.power(x, deg+1)]
    return poly

def augment_missing(tX, default=-999):
    """
    Adds a column for each initial column that represent the default (missing) data.
    Puts 1 if there's a missing value in the initial column at that position, 0 otherwise.
  
    Parameters:

    tx : input data 
    default : the value to consider as default one

    Returns:
    The new columns of 1s and 0s.
  
    """

    cols = np.zeros((tX.shape[0]))
    for i in range(tX.shape[1]):
        new_col = np.where(tX[:, i]==-999, 1,0) #if missing put 1 else 0
        tX=np.c_[tX, new_col]
    return tX


def remove_outliers(tX, std_val):
    """
    Replaces the outliers' values. We consider an outlier each value 
    that is greater than median+std_val*std or lower than median-std_val*std
  
    Parameters:

    tx : input data 
    std_val : the value that represents how far we accept a data point value to be far from the feature median median

    Returns:
    The input data with outliers replaced by reduced values.
  
    """

    std_values = np.std(tX, axis=0)
    medians = np.median(tX, axis=0)
    means = np.mean(tX, axis=0)
    tX_no_outliers = tX.copy()
    tX_bool_sup = tX.copy()
    tX_bool_inf = tX.copy()

    for j in range(tX.shape[1]):
        tX_bool_sup[:, j] = tX[:, j] > medians[j]+std_val*std_values[j]
    for j in range(tX.shape[1]):
        tX_bool_inf[:, j] = tX[:, j] < medians[j]-std_val*std_values[j]

    tX_bool_sup = tX_bool_sup.astype(bool)
    tX_bool_inf = tX_bool_inf.astype(bool)
    for j in range(tX.shape[1]):
        tX_no_outliers[:, j] = np.where(tX_bool_sup[:, j], medians[j]+2*std_values[j], tX_no_outliers[:, j])
    for j in range(tX.shape[1]):
        tX_no_outliers[:, j] = np.where(tX_bool_inf[:, j], medians[j]-2*std_values[j], tX_no_outliers[:, j])

    return tX_no_outliers

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold
  
    Parameters:

    y : output data
    k_fold: number of k-folds
    seed: seed for reproducibility

    Returns:
    k indices for k-fold
  
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def sigmoid(x):
    '''
    apply the sigmoid function on t.
    '''
    return 1 / (1+np.exp(-x))

def split_keeping_indices(tX, ids, der_poly_deg=11, pri_poly_deg=5, y=None, train=True):
    """
    Split data by PRI_jet_num values {0,1,2,3}
  
    Parameters:
    
    tX : input data
    ids : data ids
    der_poly_deg : degree of derived values
    pri_poly_deg : degree of primitive values
    y : output data
    train : True if the split concerns the training data

    Returns:
    The split data
  
    """

    nb_der_columns = 13
    pri_jet_num_ind = 1+nb_der_columns*der_poly_deg+10 #index of pri_jet_num in the indexed expanded table

    i_0 = [i for i in  range(tX.shape[0]) if tX[i,pri_jet_num_ind]==0]
    i_1 = [i for i in  range(tX.shape[0]) if tX[i,pri_jet_num_ind]==1]
    i_2 = [i for i in  range(tX.shape[0]) if tX[i,pri_jet_num_ind]==2]
    i_3 = [i for i in  range(tX.shape[0]) if tX[i,pri_jet_num_ind]==3]

    tX_0 = tX[i_0]
    tX_1 = tX[i_1]
    tX_2 = tX[i_2]
    tX_3 = tX[i_3]

    ids_0 = ids[i_0]
    ids_1 = ids[i_1]
    ids_2 = ids[i_2]
    ids_3 = ids[i_3]

    if train:
        y_0 = y[i_0]
        y_1 = y[i_1]
        y_2 = y[i_2]
        y_3 = y[i_3]
        return tX_0, y_0, tX_1, y_1, tX_2, y_2, tX_3, y_3, ids_0, ids_1, ids_2, ids_3
      
    return tX_0, tX_1, tX_2, tX_3, ids_0, ids_1, ids_2, ids_3

def reassemble_predictions_by_ids(y0, y1, y2, y3, ids0, ids1, ids2, ids3):
    """
    Concatenates the split predictions in the initial order  
  
    Parameters:
    
    y0, y1, y2, y3 : split output
    ids0, ids1, ids2, ids3 : split ids

    Returns:
    The concatenated predicitions in the right order.
  
    """

    y0 = np.c_[ids0, y0]
    y1 = np.c_[ids1, y1]
    y2 = np.c_[ids2, y2]
    y3 = np.c_[ids3, y3]

    y = np.concatenate((y0,y1,y2,y3), axis=0)
    y = y[np.argsort(y[:,0], axis=0)]

    return y[:, 1]

def replace_defaults_median(tX, default_val=-999.0):
    """
    Replaces the default values (-999) by the median of the underlying feature.
  
    Parameters:
    
    tX : input data
    default_val : the value that's considered default to be replaced

    Returns:
    The input data with default values replaced.
  
    """

    x = tX.copy()
    for j in range(x.shape[1]):
        f_values = x[:, j]
        missing_values = f_values == default_val
        median = np.median(f_values[~missing_values])
        f_values[missing_values] = median
    return x

def split_data(y, x, ratio, seed=1):
    """
    Splits the data for tests

    Parameters:
    
    x : input data
    y : output data
    ratio : the splitting ratio
    seed : seed for reproducibility

    Returns:
    Split data
  
    """
    assert 0 < ratio < 1, "0 < ratio < 1 required"

    yc = y.copy()
    xc = x.copy()

    spl = int(ratio*len(y))

    np.random.seed(seed)
    idx = np.random.permutation(len(y))
    tr_idx = idx[:spl]
    te_idx = idx[spl:]
    
    return yc[tr_idx], xc[tr_idx], yc[te_idx], xc[te_idx]

def standardize(x):
    """
    Standarize the input data

    Parameters:
    
    x : input data

    Returns:
    Standaridized input
  
    """

    mean_x = np.median(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x
    
def standardize_columns(tX, columns_not_std):
    """
    Standarize the input data, expect columns stated in columns_not_std

    Parameters:
    
    tX : input data
    columns_not_std: the columns not the standardize

    Returns:
    Standaridized input
  
    """

    columns_to_standardize = list(range(tX.shape[1]))

    for col in columns_not_std:
        columns_to_standardize.remove(col)

    for i in columns_to_standardize:
        tX[:, i] = standardize(tX[:, i])
    return tX

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse
    
def preprocess(tX, poly_value_der=11, poly_value_pri=5, columns_not_std=[22], outlier_limit=5):
    """
    The modularized cleaning and preparation function. It creates the new columns that highlight the missing values,
    replaces default values (-999) by the median, removes the values we considered as outliers,
    standardizes the chosen columns and does the polynomial expansion, and finally adds the previously mentioned columns.

    Parameters:
    
    tX : input data
    der_poly_deg : degree of derived values
    pri_poly_deg : degree of primitive values
    columns_not_std: the columns not the standardize
    outlier_limit = the value that represents how far we accept a data point value to be far from the feature median median

    Returns:
    Preprocessed data
  
    """

    pri_index = 13
    cols = augment_missing(tX)
    tX = replace_defaults_median(tX)
    tX = remove_outliers(tX, outlier_limit)
    tX = standardize_columns(tX, columns_not_std)
    tX_1 = build_poly(tX[:, :pri_index], poly_value_der)
    tX_2 = build_poly(tX[:, pri_index:], poly_value_pri)
    tX = np.c_[tX_1, tX_2]
    tX = np.c_[tX, cols]
    return tX 

def compute_binary_statistics(y_test, y_pred):
    """
    Computes the accuracy and F1 score of our predicitons.

    Parameters:
    y_test : the actual data outputs
    y_pred : predicted data outputs

    Returns:
    The accuracy and F1 score
  
    """
    "Binary classifiers statistics"

    y_pred=np.array(y_pred)
    y_test=np.array(y_test)
    
    TP = (y_pred==1) & (y_test==1)
    n_TP = np.sum(TP)
    
    FN = (y_pred==-1) & (y_test==1)
    n_FN = np.sum(FN)
    
    FP = (y_pred==1) & (y_test==-1)
    n_FP = np.sum(FP)

    TN = (y_pred==-1) & (y_test==-1)
    n_TN = np.sum(TN)
    
    acc = (n_TP+n_TN) / (n_TP+n_FP+n_FN+n_TN)
    F1 = (2.*n_TP) / (2*n_TP+n_FP+n_FN)

    return acc, F1

def find_best_lambda(tX):
    """
    Grid search to find the best lambda value for the model

    Parameters:
    tX : input data

    Returns:
    The lambda value that fits best our model
  
    """

    results = []
    lambdas1 = np.linspace(1e-1, 1e-4,20)
    lambdas2 = np.linspace(5e-4, 1e-6,20)  
    lambdas3 = np.linspace(5e-6, 1e-8,20)
    lambdas4 = np.linspace(5e-8, 1e-10,20)
    lambdas = np.concatenate((lambdas1, lambdas2, lambdas3, lambdas4), axis=0)
    best_i = 0
    best_acc = 0       
    for i in range(len(lambdas)):
        tX = preprocess(tX)
        y_tr, x_tr, y_te, x_te = split_data(y, tX, ratio=0.5)
        w, loss = ridge_regression(y_tr, x_tr, lambda_=lambdas[i])
        y_pred = predict_labels(w, x_te)
        stats = compute_binary_statistics(y_te, y_pred)
        results.append(stats[0])
        if stats[0] > best_acc:
          best_acc = stats[0]
          best_i = i
    
    return lambdas[best_i]

def compute_acc(fun, y, x, ids, k_indices, k, **args):
    """
    Computes the accuracy of the training and test set of the k-th iteration.

    Parameters:
    fun : the function to compute the weights
    y = output data
    x = input data
    ids : data ids
    k_indices : k indices for k-fold
    k : k-th iteration of cross-validation
    args : other needed arguments, depending on fun

    Returns:
    The accuracy of the training and test set of the k-th iteration
  
    """

    te_idx = k_indices[k]
    te_y=y[te_idx]
    te_x=x[te_idx]
    te_ids = ids[te_idx]

    tr_idx = np.delete(k_indices, k, axis=0).flatten()
    tr_y=y[tr_idx]
    tr_x=x[tr_idx]
    tr_ids = ids[tr_idx]

    # Data is already preprocessed

    txr0, yr0, txr1, yr1, txr2, yr2, txr3, yr3, idsx0, idsx1, idsx2, idsx3 = split_keeping_indices(tr_x, tr_ids, y=tr_y)
    txe0, txe1, txe2, txe3, ids0, ids1, ids2, ids3 = split_keeping_indices(te_x, te_ids, train=False)
  
    w0, tr_loss0 = fun(yr0, txr0, **args)
    w1, tr_loss1 = fun(yr1, txr1, **args)
    w2, tr_loss2 = fun(yr2, txr2, **args)
    w3, tr_loss3 = fun(yr3, txr3, **args)

    tr_pred0 = predict_labels(w0, txr0)
    tr_pred1 = predict_labels(w1, txr1)
    tr_pred2 = predict_labels(w2, txr2)
    tr_pred3 = predict_labels(w3, txr3)

    te_pred0 = predict_labels(w0, txe0)
    te_pred1 = predict_labels(w1, txe1)
    te_pred2 = predict_labels(w2, txe2)
    te_pred3 = predict_labels(w3, txe3)

    tr_acc0, tr_f10 = compute_binary_statistics(yr0, tr_pred0)
    tr_acc1, tr_f11 = compute_binary_statistics(yr1, tr_pred1)
    tr_acc2, tr_f12 = compute_binary_statistics(yr2, tr_pred2)
    tr_acc3, tr_f13 = compute_binary_statistics(yr3, tr_pred3)
    tr_acc = (tr_acc0+tr_acc1+tr_acc2+tr_acc3)/4

    te_acc0, te_f10 = compute_binary_statistics(ye0, te_pred0)
    te_acc1, te_f11 = compute_binary_statistics(ye1, te_pred1)
    te_acc2, te_f12 = compute_binary_statistics(ye2, te_pred2)
    te_acc3, te_f13 = compute_binary_statistics(ye3, te_pred3)
    te_acc= (te_acc0+te_acc1+te_acc2+te_acc3)/4

    return tr_acc, te_acc

def splitted_cross_validation(fun, y, x, ids, k_indices, k_fold, **args):

    """
    Cross validation used to evaluate our model

    Parameters:
    fun : the function to compute the weights
    y = output data
    x = input data
    ids : data ids
    k_indices : k indices for k-fold
    k-fold : number of subsets
    args : other needed arguments, depending on fun

    Returns:
    The mean accuracy of the training and test set.
  
    """

    tr_acc_arr = []  # accuracies of training sets
    te_acc_arr = []  # accuracies of test sets

    for k in range(k_fold):
        k_tr_acc, k_te_acc = compute_acc(fun, y, x, ids, k_indices, k, **args)
        tr_acc_arr.append(k_tr_acc)
        te_acc_arr.append(k_te_acc)

    tr_acc = np.mean(tr_acc_arr)
    te_acc = np.mean(te_acc_arr)
    return tr_acc, te_acc




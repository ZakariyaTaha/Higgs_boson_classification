#run.py : Camille Arruat, Romain Wiorowski, Taha Zakariya
#Submission ID: 164433
#This is the file to use to reproduce the predictions in AIcrowd

# Import all the needed functions
import numpy as np
from implementations import *
from costs import *
from helpers import *
from proj1_helpers import *

# Load the data

DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Data Cleanup & Preperation

tX_test = preprocess(tX_test)
tX = preprocess(tX)

'''
 Split according to the only categorical feature (PRI_jet_num), 
 keeping track on the ids the reassemble the predictions
'''

tex0, tex1, tex2, tex3, ids0, ids1, ids2, ids3 = split_keeping_indices(tX_test, ids_test, train=False)
tX_0, y_0, tX_1, y_1, tX_2, y_2, tX_3, y_3, idsx0, idsx1, idsx2, idsx3 = split_keeping_indices(tX, ids, y=y)

# Apply ridge regression on each subset
# The lambda value used for ridge regression had been computed by grid searching the best value

weights0, loss01 = ridge_regression(y_0, tX_0)
weights1, loss1 = ridge_regression(y_1, tX_1)
weights2, loss2 = ridge_regression(y_2, tX_2)
weights3, loss3 = ridge_regression(y_3, tX_3)


# Create Predictions, reassemble them in the initial order, and write to csv file for submission

y_pred0 = predict_labels(weights0, tex0)
y_pred1 = predict_labels(weights1, tex1)
y_pred2 = predict_labels(weights2, tex2)  
y_pred3 = predict_labels(weights3, tex3)

y_pred = reassemble_predictions_by_ids(y_pred0, y_pred1, y_pred2, y_pred3, ids0, ids1, ids2, ids3)

OUTPUT_PATH = '../Prediction'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------

# import the necessary packages
from tes1_back import *
from create_test_back import *
import pandas as pd
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import  cross_val_score

# variable to hold the results
output = []

h5f_data = h5py.File('Output/data.h5', 'r')
h5f_label = h5py.File('Output/labels.h5', 'r')

h5f_test_data = h5py.File('Output/test_data.h5', 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_test_features_string = h5f_test_data['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)
global_test_features = np.array(global_test_features_string)

h5f_data.close()
h5f_label.close()
h5f_test_data.close()

# verify the shape of the feature vector and labels
print(".... features shape: {}".format(global_features.shape))
print(".... labels shape: {}".format(global_labels.shape))

                                                                                         
trainDataGlobal = global_features
trainLabelsGlobal = global_labels
testDataGlobal = global_test_features

print(".... splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))

# filter all the warnings
import warnings
warnings.filterwarnings('ignore')

#-----------------------------------
# GETTING THE OUTPUT
#-----------------------------------

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=4, random_state=2,min_samples_split=2)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)
predic_ = clf.predict(testDataGlobal)
for i in predic_:
    if(i==0):
        output.append('car')
    else:
        output.append('motorbike')  

pd.DataFrame(output).to_excel('output1.xlsx', header=False, index=False)


# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:31:38 2019

@author: ritik
"""

#-----------------------------------
# FEATURE EXTRACTION
#-----------------------------------

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

# fixed-sizes for image
fixed_size = tuple((500, 500))

# path to training data
train_path = r"C:\Users\ritik\Desktop\train"



# bins for histogram
bins = 8



# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# Initialize lists to hold feature vectors and labels
global_features = []
labels = []

# num of images per class
images_per_class = 80

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir_ = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    list_of_files = os.listdir(dir_)
    #k = 1
    # loop over the images in each sub-folder
    for img_file in list_of_files:
        img_file_path = os.path.join(train_path, current_label, img_file)

        # read the image and resize it to a fixed-size
        image = cv2.imread(img_file_path)
        if image is None:
            raise RuntimeError("No image Found")
            
        image = cv2.resize(image, fixed_size)


        
        #  Feature extraction
        
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        
        # Concatenate features
        #
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print(".... processed folder: {}".format(current_label))


print(".... completed Feature Extraction of training data...")

# get the overall feature vector size
print(".... feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print(".... training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print(".... training labels encoded...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print(".... feature vector normalized...")

#print(".... target labels: {}".format(target))
print(".... target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File('Output/data.h5', 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File('Output/labels.h5', 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print(".... end of vectorisation..")

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 01:12:41 2019

@author: ritik
"""

#-----------------------------------
# FEATURE EXTRACTION OF TEST DATA
#-----------------------------------

# import the necessary packages

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

fixed_size = tuple((500, 500))

test_path = r"C:\Users\ritik\Desktop\test"


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

glob_features = []



list_of_files = os.listdir(test_path)

    # loop over the images in each sub-folder
for imag_file in list_of_files:
    imag_file_path = os.path.join(test_path,imag_file)

    # read the image and resize it to a fixed-size
    image_ = cv2.imread(imag_file_path)
    if image_ is None:
        raise RuntimeError("No image")
    
        
    image_ = cv2.resize(image_, fixed_size)
    
    #  Feature extraction
    
    fv_hu_moments = fd_hu_moments(image_)
    fv_haralick   = fd_haralick(image_)
    fv_histogram  = fd_histogram(image_)

    
    # Concatenate features
    
    glob_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    glob_features.append(glob_feature)


print(".... completed Feature Extraction of test data...")

# get the overall feature vector size
print(".... feature vector size {}".format(np.array(glob_features).shape))

    
# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features_ = scaler.fit_transform(glob_features)
print(".... feature vector normalized...")

# save the feature vector using HDF5
h5f_test_data = h5py.File('Output/test_data.h5', 'w')
h5f_test_data.create_dataset('dataset_1', data=np.array(rescaled_features_))

h5f_test_data.close()

print(".... end of vectorisation of test data..")






















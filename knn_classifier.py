# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:04:51 2020

@author: Brian
"""


# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import pandas as pd


def image_to_feature_vector(image, size = (32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins = (8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()

path = "dogs-vs-cats"
k = 201
jobs = -1
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(path))
#args["dataset"])
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []



cc = pd.read_csv('C:/Users/Brian/Desktop/MeerkatAI-KNN-Valuation-Model/sofa.csv')
price = list(cc['price'])



# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = price[int(imagePath.split(os.path.sep)[-1].split(".")[0]) - 1]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))
        
        
# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(features.nbytes / (1024 * 1000.0)))


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size = 0.25, random_state = 42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size = 0.25, random_state = 42)


# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
#model = KNeighborsClassifier(n_neighbors= args["neighbors"],
#	n_jobs= args["jobs"])
model = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree', n_jobs = jobs)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


'''
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors = k, algorithm = 'kd_tree').fit(X)   
distances, indices = nbrs.kneighbors(X)
indices
'''

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
#model = KNeighborsClassifier(n_neighbors=5'args["neighbors"]',
#	n_jobs=-1'args["jobs"]')
model = KNeighborsClassifier(n_neighbors=k, n_jobs= jobs)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))
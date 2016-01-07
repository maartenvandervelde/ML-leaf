#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
import sys, os

# Generates a Bag-of-Words histogram of an image, given a text file containing the cluster centers (visual words) and a text file containing the image's features.
# The resulting histogram is normalised and saved in a new text file called [image name]_K[number of clusters]_hist.txt.
#
# Example call: "python generateHistogram.py kmeans_centers_N539_K3.txt 10874.txt".
#
# 07 Jan 2016 Maarten


class GenerateHistogram():	
		
	def __init__(self, clusters_file, dim=128):
		self.dim = dim
		self.clusters = self.read_cluster_centers(clusters_file)
		self.K = len(self.clusters)
		
	
	# Read cluster centers from file
	def read_cluster_centers(self, clusters_file):
		dat = np.fromfile(clusters_file, sep=" ")
		clusters = np.reshape(dat, (-1, self.dim))
		return clusters

	# Generate histogram for image
	def generate_histogram(self, img_file):

		# Read image's features
		img_features = self.read_image_features(img_file)
		
		# Find Euclidean distance of each feature to each cluster center
		distances = cdist(img_features, self.clusters, metric="euclidean")
		
		# Each feature is assigned to its nearest cluster (visual word)
		words = np.argmin(distances, axis = 1)
		
		# Store word frequency in a histogram of length K
		histogram = np.bincount(words)
		
		# Normalise histogram so all values are between 0 and 1
		histogram = histogram.astype(float) / histogram.max()

		# Save histogram to file
		filename = os.path.splitext(img_file)[0] + '_K' + str(self.K) + '_hist.txt'
		np.savetxt(filename, histogram)
		print 'Histogram written to', filename

	
	# Read image features from file
	def read_image_features(self, img_file):
		dat = np.fromfile(img_file, sep=" ")
		img_features = np.reshape(dat, (-1, self.dim))
		return img_features

	
if __name__ == '__main__':
	clusters_file = sys.argv[1]
	img_feats = sys.argv[2]
	genHist = GenerateHistogram(clusters_file)
	genHist.generate_histogram(img_feats)

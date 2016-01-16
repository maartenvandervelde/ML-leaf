#!/usr/bin/env python

# Runs through all scripts related to Bag of Words.
# Given a directory of training *.fts files, clusters the features, and uses these clusters to generate a histogram for each training and testing file.
#
#
#
# Arguments:
# 	- train_dir (required)	: directory containing training *.fts files.
# 	- test_dir (required)	: directory containing testing *.fts files.
#	- N_fts (required)		: number of *.fts files to use for clustering
#	- K (required)			: number of clusters in K-Means clustering
#
#
#
# Example call:
#	python bagOfWords.py imageclef/train imageclef/test 100 100
#
#	(Combines 100 randomly selected *.fts files, yielding ~10.000 SIFT features, which are clustered into 100 clusters (this can take several minutes!).
#	Histograms with length 100 are generated for each *.fts file in the training and testing directories.)
#
#

import sys

import concatenateFeatures
import kmeans
import generateHistogram

def main(argv):
	argc = len(argv)
	
	if (argc is not 5):
		print '\nWarning: incorrect argument(s) to bagOfWords.py. Expected arguments:\n\n' \
		'\t- train_dir (required)\t: directory containing training *.fts files.\n' \
		'\t- test_dir (required)\t: directory containing testing *.fts files.\n' \
		'\t- N_fts (required)\t: number of *.fts files to use for clustering.\n'\
		'\t- K (required)\t\t: number of clusters in K-Means clustering\n'
		sys.exit(1)
	
	train_dir = argv[1]
	test_dir = argv[2]
	n_fts = int(argv[3])
	n_clusters = int(argv[4])
	
	# 1. Combine files containing SIFT features of individual images into a single file for clustering.
	print '\n Preparing for clustering...\n'
	concatFile = concatenateFeatures.main([train_dir, n_fts])
	
	# 2. Cluster features into K clusters with k-means.
	print '\n Clustering...\n'
	centersFile = kmeans.main([concatFile, n_clusters])
	
	# 3. Generate histograms for each training file using the cluster centers.
	print '\n Generating histograms for training files...\n'
	generateHistogram.main([centersFile, train_dir])
	
	# 4. Generate histograms for each testing file using the cluster centers.
	print '\n Generating histograms for testing files...\n'
	generateHistogram.main([centersFile, test_dir])
	
	print '\n Bag of Words done!\n'

if __name__ == '__main__':
	main(sys.argv)
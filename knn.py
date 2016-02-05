#!/usr/bin/env python

import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter
import sys, os, glob, csv

# Classifies *.hst files of leaves using the K-Nearest-Neighbour algorithm.
#
# Arguments:
# 	- train_dir (required)	: directory containing training *.hst files.
#	- test_dir (required)	: directory containing testing *.hst files.
#	- max_K (required)		: KNN classification will report accuracy for K = 1 up to K = max_K.
#	- T (required)			: Classification is considered correct if answer appears in top T results.
#
# Example call: "python knn.py train test 3 5".
#
#

class KnnClassifier():	
		
	def __init__(self, traindir, K):
		self.K = K
		
		# Load all histograms from training directory
		self.data, self.classes = self.load_training_data(traindir)
		
		n_items, n_dim = self.data.shape
		n_classes = len(set(self.classes))
		
		print 'Performing K-Nearest-Neighbour classification in {} dimensions using {} training items from {} classes.'.format(n_dim, n_items, n_classes)
	
	# Returns all histograms from training directory in a numpy array together with a list of the corresponding classes.
	def load_training_data(self, traindir):
	
		trainfiles = get_filenames(traindir)

		data = []
		classes = []
		
		for file in trainfiles:
			leafTypeString = get_leaftype(os.path.splitext(file)[0]) 

			if (leafTypeString is not None):
				data.append(np.loadtxt(traindir + '/' + file))
				classes.append(leafTypeString)
			else:
				print 'Warning! Unlabeled item in train data, will be ignored: ' + file

		data = np.array(data)
		
		return data, classes
	
	
	def classify(self, testdir):
		testfiles = get_filenames(testdir)
		
		print 'Classifying {} test items...'.format(len(testfiles))
		
		correct = 0
		count = 0
		
		classifications = []
		
		for file in testfiles:
			leafTypeString = get_leaftype(os.path.splitext(file)[0]) 

			if (leafTypeString is not None):
				count += 1
				
				# Read histogram
				test_item = np.atleast_2d(np.loadtxt(testdir + '/' + file))
				
				# Find class votes of K nearest neighbours
				distances = cdist(self.data, test_item, metric="euclidean")
				distances = np.column_stack([distances, self.classes])
				distances_sorted = distances[np.argsort(distances[:,0])]
				
				class_votes = distances_sorted[0:self.K,-1]
				
				# Store votes together with the actual class
				classifications.append(np.insert(class_votes, 0, leafTypeString))
				
			else:
				print 'Warning! Unlabeled item in test data, will be ignored: ' + file
		
		return np.array(classifications)
		
		
	# Calculates and prints accuracy for a range of K's, given an array of classifications
	def print_accuracy(self, classifications, lower_K, upper_K, T=1):
		count = len(classifications)
		accuracies = []
		
		for i in xrange(lower_K + 1, upper_K + 2):
			correct = 0
			
			for row in classifications:
				class_real = row[0]

				# Assign T classes to item using the i nearest neighbours' votes 
				class_vote = Counter(row[1:i]).most_common(T)
				for item in class_vote:
					if class_real in item:
						correct += 1
			
			accuracy = float(correct) * 100 / count
			accuracies.append(accuracy)
			
			print 'Accuracy at K = {}: {}%.'.format(i-1, accuracy)
		
		# Report highest accuracy
		max_ind = np.argmax(accuracies)
		max_acc = accuracies[max_ind]
		print 'Highest classification accuracy is {}% at K = {}.'.format(max_acc, max_ind + 1)
			
#Get all *.hst in a specified folder
def get_filenames(dir):
	files = []
	files.extend(glob.glob1(dir, '*.hst'))
	return files
	
# Returns the class of a leaf given its number.
def get_leaftype(imagename):
	f = open('imagetable.csv')
	csv_f = csv.reader(f)
	for row in csv_f:
		if row[0] == imagename:
			return row.pop()


def main(argv):
	argc = len(argv)
	
	if (argc is not 4):
		print '\nWarning: incorrect argument(s) to knn.py. Expected arguments:\n\n' \
		'\t- train_dir (required)\t: directory containing training *.hst files.\n' \
		'\t- test_dir (required)\t: directory containing testing *.hst files.\n'\
		'\t- max_K (required)\t: KNN classification will report accuracy for K = 1 up to K = max_K.\n'\
		'\t- T (required)\t\t: Classification is considered correct if answer appears in top T results.\n'
		sys.exit(1)
	
	traindir = argv[0]
	testdir = argv[1]
	K = int(argv[2])
	T = int(argv[3])
	
	# Load training data
	KNN = KnnClassifier(traindir, K)
	
	# Classify test data
	classifications = KNN.classify(testdir)
	
	# Report accuracy
	KNN.print_accuracy(classifications, 1, K, T)
		
if __name__ == '__main__':
	main(sys.argv[1:])

		

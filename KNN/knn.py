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
#	- K (required)			: number of neighbours used in KNN classification.
#
# Example call: "python knn.py train test 3".
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
		
		correct = 0
		count = 0
		
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
				
				# Assign majority class to test item
				test_item_class = Counter(class_votes).most_common(1)[0][0]
				
				# Check correctness
				if (leafTypeString == test_item_class):
					correct += 1
				#else:
					#print file + ' classified incorrectly, as ' + test_item_class + ' instead of ' + leafTypeString				
			else:
				print 'Warning! Unlabeled item in test data, will be ignored: ' + file
		
		print 'Classification of {} test items finished with accuracy of {}%.'.format(count, float(correct)*100/count)
		
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
	
	if (argc is not 3):
		print '\nWarning: incorrect argument(s) to knn.py. Expected arguments:\n\n' \
		'\t- train_dir (required)\t: directory containing training *.hst files.\n' \
		'\t- test_dir (required)\t: directory containing testing *.hst files.\n'\
		'\t- K (required)\t\t: number of neighbours used in KNN classification.\n'
		sys.exit(1)
	
	traindir = argv[0]
	testdir = argv[1]
	K = int(argv[2])
	
	KNN = KnnClassifier(traindir, K)
	
	KNN.classify(testdir)
		
if __name__ == '__main__':
	main(sys.argv[1:])

		

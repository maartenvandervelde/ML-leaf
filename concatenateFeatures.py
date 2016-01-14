#!/usr/bin/env python

import sys, os, glob, random

# Combines the SIFT feature files (*.fts) in the specified directory into a single file that can be used for k-means clustering.
# Arguments:
# 	- directory (required)
#	- N (optional): the number of files to be randomly sampled from directory. Omit to use all files in directory.
#
# Example call: "python concatenateFeatures.py grootste5 100". This will concatenate a random selection of 100 files in the directory 'grootste5'.
#
	
def main(argv):
	argc = len(argv)
	
	directory = argv[0]
	
	files = []
	files.extend(glob.glob1(directory, '*.fts'))
	
	# Use subset of all files if specified, otherwise use all files
	num_files = len(files)
	
	if (argc == 2 and int(argv[1]) <= num_files):
		num_files = int(argv[1])
	
	files_sample = random.sample(files, num_files)
	
	allFeatures = ''
	
	for file in files_sample:
		filepath = directory + '/' + file
		
		with open(filepath, 'r') as f:
			fileFeatures = f.read()
		
		allFeatures += fileFeatures + ' '

	
	concatFile = directory.replace('/','-') + '_concat_' + str(num_files) + '.fts'
	with open(concatFile, 'w') as f:
		f.write(allFeatures)
	
	print 'Concatenated features from ' + str(num_files) + ' files in the directory ' + directory + ' into the file ' + concatFile
	
	return concatFile
	
if __name__ == '__main__':
	main(sys.argv[1:])
	

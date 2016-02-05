#!/usr/bin/env python


# Clusters SIFT descriptors using K-Means clustering.
# The resulting cluster centres are written to a file.
#
# Arguments:
# 	- ftsfile (required)	: file containing a list of 128-element SIFT descriptors.
#
# Example call: "python kmeans.py features.fts".
#
#

import numpy as np
import matplotlib.pyplot as plt
import random, sys, time

class KMeans():
	def __init__(self, K, file, dim=128):
		self.K = K
		self.dim = dim
		self.X = self.read_keypoints(file)
		self.X = (self.X / float(np.amax(self.X))) * 2 - 1
		self.N = len(self.X)
		
		self.means = None
		self.clusters = None
		
		print "Clustering", str(len(self.X)), "keypoints from the file", file, "into", str(self.K), "clusters..."

 
	def read_keypoints(self, file):
		dat = np.fromfile(file, sep=" ")
		keypoints = np.reshape(dat, (-1, self.dim))
		return keypoints

	def _cluster_points(self):
		means = self.means
		clusters  = {}
		for keypoint in self.X:
			bestmeanskey = min([(i[0], np.linalg.norm(keypoint-means[i[0]])) for i in enumerate(means)], key=lambda t:t[1])[0]
			try:
				clusters[bestmeanskey].append(keypoint)
			except KeyError:
				clusters[bestmeanskey] = [keypoint]
		self.clusters = clusters
 
	def _reevaluate_centers(self):
		clusters = self.clusters
		newmeans = []
		keys = sorted(self.clusters.keys())
		for k in keys:
			newmeans.append(np.mean(clusters[k], axis = 0))
		self.means = newmeans
 
	def _has_converged(self):
		K = len(self.oldmeans)
		return(set([tuple(a) for a in self.means]) == set([tuple(a) for a in self.oldmeans]) and len(set([tuple(a) for a in self.means])) == K)
 
	def find_centers(self):
		X = self.X
		K = self.K
		self.oldmeans = random.sample(X, K)
		self.means = random.sample(X, K)
		
		while not self._has_converged():
			self.oldmeans = self.means
			# Assign all points in X to clusters
			self._cluster_points()
			# Reevaluate centers
			self._reevaluate_centers()
	
	def write_centers(self):
		filename = 'kmeans_centers_N' + str(self.N) + '_K' + str(self.K) + '.txt'
		np.savetxt(filename, self.means)
		print 'Cluster centers written to', filename
		return filename

def main(argv):
	file = argv[0]
	k = int(argv[1])
	t_start = time.time()
	kmeans = KMeans(k, file)
	kmeans.find_centers()
	centers_file = kmeans.write_centers()
	t_end = time.time()
	t_elapsed = t_end - t_start
	print "Time elapsed during K-Means clustering: " + str(t_elapsed) + " seconds."	
	return centers_file
	
if __name__ == '__main__':
	main(sys.argv[1:])

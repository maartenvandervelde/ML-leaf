#!/usr/bin/env python


# K-means code adapted from https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/

import numpy as np
import matplotlib.pyplot as plt
import random, sys

class KMeans():
	def __init__(self, K, file, dim=128):
		self.K = K
		self.dim = dim
		self.X = self.read_keypoints(file)
		self.X = (self.X / float(np.amax(self.X))) * 2 - 1
		self.N = len(self.X)
		
		self.mu = None
		self.clusters = None
		
		print "Clustering", str(len(self.X)), "keypoints from the file", file, "into", str(self.K), "clusters..."

 
	def read_keypoints(self, file):
		dat = np.fromfile(file, sep=" ")
		keypoints = np.reshape(dat, (-1, self.dim))
		return keypoints

	def plot_board(self):
		X = self.X[:,0:2]
		fig = plt.figure(figsize=(5,5))
		plt.xlim(-1,1)
		plt.ylim(-1,1)
		if self.mu and self.clusters:
			mu = self.mu
			clus = self.clusters
			K = self.K
			for m, clu in clus.items():
				cs = plt.cm.spectral(1.*m/self.K)
				plt.plot(zip(*clus[m])[0], zip(*clus[m])[1], '.', markersize=8, color=cs, alpha=0.5)
				plt.plot(mu[m][0], mu[m][1], 'o', marker='*', markersize=12, color=cs)
		else:
			plt.plot(zip(*X)[0], zip(*X)[1], '.', alpha=0.5)
		title = 'K-means with random initialization'
		pars = 'N=%s, K=%s' % (str(self.N), str(self.K))
		plt.title('\n'.join([pars, title]), fontsize=16)
		plt.savefig('kmeans_N%s_K%s.png' % (str(self.N), str(self.K)), bbox_inches='tight', dpi=200)
 
	def _cluster_points(self):
		mu = self.mu
		clusters  = {}
		for x in self.X:
			bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) for i in enumerate(mu)], key=lambda t:t[1])[0]
			try:
				clusters[bestmukey].append(x)
			except KeyError:
				clusters[bestmukey] = [x]
		self.clusters = clusters
 
	def _reevaluate_centers(self):
		clusters = self.clusters
		newmu = []
		keys = sorted(self.clusters.keys())
		for k in keys:
			newmu.append(np.mean(clusters[k], axis = 0))
		self.mu = newmu
 
	def _has_converged(self):
		K = len(self.oldmu)
		return(set([tuple(a) for a in self.mu]) == set([tuple(a) for a in self.oldmu]) and len(set([tuple(a) for a in self.mu])) == K)
 
	def find_centers(self):
		X = self.X
		K = self.K
		self.oldmu = random.sample(X, K)
		self.mu = random.sample(X, K)
		while not self._has_converged():
			self.oldmu = self.mu
			# Assign all points in X to clusters
			self._cluster_points()
			# Reevaluate centers
			self._reevaluate_centers()
	
	def write_centers(self):
		filename = 'kmeans_centers_N' + str(self.N) + '_K' + str(self.K) + '.txt'
		np.savetxt(filename, self.mu)
		print 'Cluster centers written to', filename
		
	
if __name__ == '__main__':
	file = sys.argv[1]
	k = int(sys.argv[2])
	kmeans = KMeans(k, file)
	kmeans.find_centers()
	kmeans.write_centers()
	kmeans.plot_board()

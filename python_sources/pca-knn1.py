#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import neighbors
import datetime
import random
from sklearn.decomposition import PCA


class KNN:
	def __init__(self):
		self.n_neighbors = 3

	def load_data(self):

		train = np.array(pd.read_csv("../input/train.csv"))
		test = np.array(pd.read_csv("../input/test.csv"))		
		self.trainX = train[:,1:]
		self.trainY = train[:,0]
		self.testX = test

		pca = PCA(n_components=35,whiten=True)
		self.trainX = pca.fit_transform(self.trainX)
		self.testX = pca.transform(self.testX)

		# train = np.array(pd.read_csv("../data/train.csv"))
		# rows = train.shape[0]
		# idxs = range(rows)
		# random.shuffle(idxs)
		# idxs = np.array(idxs)
		# train_idxs = idxs[0:30000]
		# test_idxs = idxs[30000:32000]

		# self.trainX = train[train_idxs,1:]
		# self.trainY = train[train_idxs,0]

		# self.testX = train[test_idxs,1:]
		# self.testY = train[test_idxs,0]


	def train_model(self):
		clf = neighbors.KNeighborsClassifier(self.n_neighbors,  weights='uniform',  algorithm='auto')
		self.clf = clf.fit(self.trainX, self.trainY)


	def predict(self):
		sample_num = self.testX.shape[0]
		result = []
		for idx in range(sample_num):
			res = self.clf.predict(self.testX[idx,:].reshape(1,-1))
			result.append(res)
			if idx % 1000 == 0:
				print (idx)
		self.predictY = np.array(result)

	def measure(self):
		# print self.predictY
		self.testY = self.testY.reshape(-1,1)
		# print self.testY
		accuracy = sum(self.predictY == self.testY) * 1.0/self.testX.shape[0]

	def run(self):
		self.load_data()
		self.train_model()
		self.predict()
		# self.measure()
		return self.predictY


start_time = datetime.datetime.now()

knn = KNN()
predictY = knn.run()


np.savetxt('submission_pca_knn1.csv', np.c_[range(1,len(predictY)+1),predictY], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


end_time = datetime.datetime.now()








#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import neighbors
import datetime
import random
from sklearn.decomposition import PCA
from sklearn.svm import SVC




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
		print(pca.explained_variance_ratio_)
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





	def run(self):
		self.load_data()



start_time = datetime.datetime.now()

knn = KNN()
knn.run()


# np.savetxt('submission_pca_svm1.csv', np.c_[range(1,len(predictY)+1),predictY], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


end_time = datetime.datetime.now()








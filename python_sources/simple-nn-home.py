from multiprocessing import Pool

import time
import timeit
from datetime import datetime, timedelta

import csv

import sys



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier



def test_settings(d):

	#layers = d[0] 
	activation_function = d[0] 
	alph = d[1]
	test_data = d[2] 
	train_data = d[3] 
	train_label = d[4]
	test_label = d[5]
	solv = d[6]

	#ts("started: " + str(activation_function) + " " + str(alph) )
	#return
	
	#mlp_classifier = MLPClassifier(hidden_layer_sizes= layers, alpha=alph , activation = activation_function, verbose=False)
	mlp_classifier = MLPClassifier(hidden_layer_sizes=(50000,5000),solver = solv,alpha=float(alph) , activation = activation_function, verbose=False)
	mlp_classifier = mlp_classifier.fit(train_data, train_label)
	mlp_prediction = mlp_classifier.predict(test_data)
	accuracy = accuracy_score(test_label, mlp_prediction)*100
	data ='{solver},{activation_function},{alpha},{accuracy}%'.format(solver=solv,activation_function = activation_function,alpha=alph,accuracy = accuracy)
    
	# 1. test alpha & activation function
	# 2. improve layers

	ts("result: " + data)


# print something with timestamp
def ts(s):
	now = datetime.now()
	print(now.strftime("%Y-%m-%d %H:%M:%S") + " // " + s)



def process_file(f):

	f = f.replace("new_zetz_","hist_")
	f = "../input/z-forecast/" + f
	train = pd.read_csv(f) # try that before we do cuts

	for r in range(0,len(train['z0']),1):
		v = int(train['z0'][r]/10)
		train['z0'][r] = v

	spl = int(len(train)/10) * 8 # that is 20% for test
	train_data = train.values[:, 1:spl] #all before
	test_data = train.values[:, spl+1:] # the remaining 20%
	
	print(f, spl, len(train), len(train_data), len(test_data))
	
	train_label = train.values[:, 0]
	test_label = train_label
	
	alphas = [0.05] # [0.0005, 0.005, 0.05, 0.5, 0.0001, 0.001, 0.01, 0.1]
	afunc = ["identity"]#, "logistic", "tanh", "relu"]
	solvers = ["lbfgs"]#, "sgd", "adam"]
	layers = [(100000,10000)] # (1000, )

	
	d = []
	for a in alphas:
		for af in afunc:
			for s in solvers:
				#for l in layers:
				d.append( [af,a,test_data, train_data, train_label, test_label, s] )

	ts("started pool")
	with Pool() as p: 
		p.map(test_settings, d)
	ts("finished")

if __name__ == "__main__":
	files = ["new_zetz_NZDUSD1.csv","new_zetz_EURGBP1.csv","new_zetz_GBPAUD1.csv","new_zetz_GBPCAD1.csv","new_zetz_GBPCHF1.csv","new_zetz_GBPJPY1.csv","new_zetz_GBPNZD1.csv","new_zetz_GBPUSD1.csv","new_zetz_AUDNZD1.csv","new_zetz_EURNZD1.csv","new_zetz_NZDCAD1.csv","new_zetz_NZDCHF1.csv","new_zetz_NZDJPY1.csv"]
	for f in files:
		process_file(f)
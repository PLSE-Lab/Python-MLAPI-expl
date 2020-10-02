# See "results" for prediction output
# See *.png for 3d visualization of classifier


import sqlite3
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import tensorflow as tf
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plot
import os
import sys

tf.logging.set_verbosity(tf.logging.ERROR)

def read_dataset():
	database_path = "../input/database.sqlite"
	conn = sqlite3.connect(database_path)
	cursor = conn.cursor()

	data = []
	for row in cursor.execute("SELECT * FROM Iris"):
		data.append(row)

	conn.close()

	return data

def label_to_id(label):
	if 'setosa' in label:
		return 0
	elif 'versicolor' in label:
		return 1
	elif 'virginica' in label:
		return 2
	else:
		print("Unkown label: " + label)
		return -1

def id_to_label(id):
	labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

	if id >= 0 and id < len(labels):
		return labels[id]
	else:
		return "Unkown label for id {0}".format(id)

def initialize_data(dataset):
	X = []
	Y = []

	for entry in dataset:
		X.append([ 1 / entry[1], 1 / entry[2], 1 / entry[3], 1 / entry[4]])

		id = label_to_id(entry[5])

		output = [0., 0., 0.]
		if id >= 0 and id < len(output):
			output[id] = 1.

		Y.append(output)

	return X, Y

X = [] #input
Y = [] #labels
dataset = read_dataset()
X, Y = initialize_data(dataset)

### PLOTTING ####
fig_1 = plot.figure(1) #Sepal length/width & species
fig_2 = plot.figure(2) #Sepal length/width & prediction
fig_3 = plot.figure(3) #Sepal length/width merged
fig_4 = plot.figure(4) #Sepal length/width/petal length & specie (color)

plot_ground_truth = fig_1.add_subplot(111, projection='3d')
plot_prediction	  = fig_2.add_subplot(111, projection='3d')
plot_merged 	  = fig_3.add_subplot(111, projection='3d')
plot_sepal_petal  = fig_4.add_subplot(111, projection='3d')

plot_ground_truth_sepal_length = []
plot_ground_truth_sepal_width  = []
plot_ground_truth_specie	   = []
plot_prediction_specie		   = []

plot_setosa_sepal_length	       = []
plot_setosa_sepal_width  	       = []
plot_setosa_petal_length	       = []

plot_versicolor_sepal_length	   = []
plot_versicolor_sepal_width  	   = []
plot_versicolor_petal_length	   = []

plot_virginica_sepal_length	       = []
plot_virginica_sepal_width  	   = []
plot_virginica_petal_length	       = []

for entry in dataset:
	plot_ground_truth_sepal_length.append(entry[1])
	plot_ground_truth_sepal_width.append(entry[2])
	plot_ground_truth_specie.append(label_to_id(entry[5]))

	spec = label_to_id(entry[5])
	if spec == 0:
		plot_setosa_sepal_length.append(entry[1])
		plot_setosa_sepal_width.append(entry[2])
		plot_setosa_petal_length.append(entry[3])
	elif spec == 1:
		plot_versicolor_sepal_length.append(entry[1])
		plot_versicolor_sepal_width.append(entry[2])
		plot_versicolor_petal_length.append(entry[3])
	elif spec == 2:
		plot_virginica_sepal_length.append(entry[1])
		plot_virginica_sepal_width.append(entry[2])
		plot_virginica_petal_length.append(entry[3])



### TF CLASSIFIER ###
tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
network = input_data(shape=[None, 4])
network = fully_connected(network, 100, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 100, activation='sigmoid', weights_init=tnorm)
network = fully_connected(network, 3, activation='sigmoid', weights_init=tnorm)

learning_rate = 0.5
regr = regression(network, optimizer='sgd', loss='mean_square',
							learning_rate = learning_rate)
model = tflearn.DNN(regr)

#model.load("iris_classifier.tfm")
model.fit(X, Y, n_epoch=2000, snapshot_epoch=False)
model.save('iris_classifier.tfm')

print("Model trained.")

prediction = model.predict(X)

i = 0
errors = 0
log = open("results", "w")

for p in prediction:
    q = p.tolist()
	max_value = max(q)
	index = q.index(max_value)
	p = [float(i)/sum(p) for i in p] #Probability distribution

	ground = Y[i]
	ground_max = max(ground)
	ground_index = ground.index(ground_max)
	if ground_index != index:
		errors = errors + 1

	plot_prediction_specie.append(index)

	print("{0}) Prediction: {1} (prob. {2:.3f}) in ".format(i, index, p[index]), ground, end = '')
	print("\tSpecie predicted: {0} ({1})".format(id_to_label(index), id_to_label(ground_index)))
	
	log.write("{0}) Prediction: {1} (prob. {2:.3f}) in ".format(i, index, p[index]))
	log.write("[{0}, {1}, {2}]".format(ground[0], ground[1], ground[2]))
	log.write("\tSpecie predicted: {0} ({1})\n".format(id_to_label(index), id_to_label(ground_index)))
	
	i = i + 1

print("Errors: {0}/{1}".format(errors, i))
log.write("Errors: {0}/{1}\n".format(errors, i))
log.close()

plot_ground_truth.scatter(plot_ground_truth_sepal_length, plot_ground_truth_sepal_width, plot_ground_truth_specie, c='r', marker='o')
plot_ground_truth.set_xlabel("Sepal Length (cm)")
plot_ground_truth.set_ylabel("Sepal Width (cm)")
plot_ground_truth.set_zlabel("Iris Specie (Ground)")
plot_ground_truth.view_init(azim=10, elev=15)

plot_prediction.scatter(plot_ground_truth_sepal_length, plot_ground_truth_sepal_width, plot_prediction_specie, c='g', marker='o')
plot_prediction.set_xlabel("Sepal Length (cm)")
plot_prediction.set_ylabel("Sepal Width (cm)")
plot_prediction.set_zlabel("Iris Specie (Predicted)")
plot_prediction.view_init(azim=10, elev=15)

plot_merged.scatter(plot_ground_truth_sepal_length, plot_ground_truth_sepal_width, plot_ground_truth_specie, c='r', marker='o')
plot_merged.scatter(plot_ground_truth_sepal_length, plot_ground_truth_sepal_width, plot_prediction_specie, c='g', marker='o')
plot_merged.set_xlabel("Sepal Length (cm)")
plot_merged.set_ylabel("Sepal Width (cm)")
plot_merged.set_zlabel("Iris Specie")
red_patch = mpatches.Patch(color='red', label='Ground Truth')
green_patch = mpatches.Patch(color='green', label='Prediction')
plot_merged.legend(handles=[red_patch, green_patch])
plot_merged.view_init(azim=10, elev=15)


plot_sepal_petal.scatter(plot_setosa_sepal_width, plot_setosa_sepal_length, plot_setosa_petal_length, c = 'r', marker='o')
plot_sepal_petal.scatter(plot_versicolor_sepal_width, plot_versicolor_sepal_length, plot_versicolor_petal_length, c = 'g', marker='o')
plot_sepal_petal.scatter(plot_virginica_sepal_width, plot_virginica_sepal_length, plot_virginica_petal_length, c = 'b', marker='o')
plot_sepal_petal.set_xlabel("Sepal Length (cm)")
plot_sepal_petal.set_ylabel("Sepal Width (cm)")
plot_sepal_petal.set_zlabel("Petal Length (cm)")
r_patch = mpatches.Patch(color='red', label='Iris-Setosa')
g_patch = mpatches.Patch(color='green', label='Iris-Versicolor')
b_patch = mpatches.Patch(color='blue', label='Iris-Verginica')
plot_sepal_petal.legend(handles=[r_patch, g_patch, b_patch])
plot_sepal_petal.view_init(azim=10, elev=15)
#plot.show()

fig_1.savefig("ground_truth.png")
fig_2.savefig("prediction.png")
fig_3.savefig("merged.png")
fig_4.savefig("sepal_petal.png")
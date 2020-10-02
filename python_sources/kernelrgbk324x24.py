import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from glob import glob 
from tensorflow.keras.preprocessing import image
import pathlib

print(tf.__version__)

#tf.config.set_visible_devices([], 'GPU')

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_WIDTH = 24
IMG_HEIGHT = 24

train_root_folder = '/kaggle/input/ffml-dataset/images/'
	  
def get_labels():
	for (root,dirs,files) in os.walk(train_root_folder):
		return dirs
	#return class_labels

def get_num_files(labels):
	num_files = 0
	for folder in labels:
		#print(train_root_folder + folder)
		for (root, dirs, files) in os.walk(train_root_folder + folder + '/'):
			num_files += len(files)
	return num_files
	

data_dir = pathlib.Path(train_root_folder)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

def decode_img(img):
	# convert the compressed string to a 3D uint8 tensor
	img = tf.image.decode_jpeg(img, channels = 3)
	#img = tf.image.rgb_to_grayscale(img)
	# Use `convert_image_dtype` to convert to floats in the [0,1] range.
	img = tf.image.convert_image_dtype(img, tf.float32)
	img = tf.image.resize(img, size = [IMG_WIDTH, IMG_HEIGHT]) 
	img = tf.reshape(img, [IMG_WIDTH, IMG_HEIGHT, 3])
	img = np.array(img)
	return img
  
def get_label(file_path):
	# convert the path to a list of path components
	parts = tf.strings.split(file_path, os.path.sep)
	# The second to last is the class-directory
	#print(parts)
	label_index = np.where(parts[-2] == CLASS_NAMES)[0][0]
	return label_index 
  
def process_path(file_path):
	label_index = get_label(file_path)
	# load the raw data from the file as a string
	img = tf.io.read_file(file_path)
	img = decode_img(img)
	return img, label_index
	
def get_training_test(labels):
	train_images = []
	train_labels = []
	test_images = []
	test_labels = []
	
	num_training = 0
	num_test = 0

	num_files = 0
	for folder in labels:
		#print(train_root_folder + folder)
		for (root, dirs, files) in os.walk(train_root_folder + folder + '/'):
			#print(files)
			i = 0
			for file in files:
				image, label = process_path(train_root_folder + folder + str('/') + file)
				if i <= 1:
					train_images.append(image)
					train_labels.append(label)
					num_training += 1
					i += 1
				else:
					test_images.append(image)
					test_labels.append(label)
					num_test += 1
			
	return num_training, num_test, train_images, train_labels, test_images, test_labels
	
def compute_distance_between_pictures(pic1, pic2):
	#print("pic1.shape = ", pic1.shape)
	distance = 0
	for i in range(0, IMG_HEIGHT):
		for j in range(0, IMG_WIDTH):
			for k in range(0, 3):
				distance += abs(pic1[i][j][k] - pic2[i][j][k])
	return distance
	
def main():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

	labels_as_string = get_labels()
	num_classes = len(labels_as_string)
	print("num_classes = ", num_classes)
	num_data = get_num_files(labels_as_string)
	print('num data=' + str(num_data))

	num_training, num_test, train_images, train_labels, test_images, test_labels = get_training_test(labels_as_string)
	print("num_training = ", num_training)
	print("num_test=", num_test)
	
	count_correct = 0 #num correctly classified
	k = 3
	for index_test in range(0, num_test):
		#take each image from test
		distance_array = []
		
		for index_train in range(0, num_training):
			distance = compute_distance_between_pictures(test_images[index_test], train_images[index_train])
			distance_array.append(distance)
		#print("distance_array.shape = ", distance_array.shape)
		#print("distance_array = ", distance_array)	
		#now sort
		index_sort = np.argsort(distance_array)[:k]
		print("index_sort = ", index_sort)
		#take the first k classes
		nearest_class = []
		for t in range(0, k):
			nearest_class.append(train_labels[index_sort[t]])
			#print(distance_array[index_sort[t]])
			
		#print("nearest_class = ", nearest_class)	
		count_repetitions = np.bincount(nearest_class)
		#print("count_repetitions", count_repetitions)
		predicted_class = np.argmax(count_repetitions)
		print("real class = ", test_labels[index_test], "predicted = ", predicted_class)
		count_correct += test_labels[index_test] == predicted_class
		print("count_correct = ", count_correct)
	print("count_correct = ", count_correct)
	print("accuracy = ", count_correct / num_test)	
	
if __name__ == '__main__':
	main()
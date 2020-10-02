# **Title: Condensing the Pixel Data into Smaller Grids**
# 
# **Author: Alex Caron**
# 
# I will investigate whether condensing the data into smaller grids (14x14, 7x7) makes a decision tree any more effective or any faster (without sacrificing accuracy). To do so, I will combine neighboring pixels into a single average pixel value: first in 2x2 groups, then in 4x4 groups.
# 
# This is also a nice exercise in working with the pandas dataframe structure.
import numpy as np   #linear algebra
import pandas as pd  #data processing
import matplotlib.pyplot as plt #data visualization
import sklearn.tree as skl #machine learning

# Helper function that provides a list of integers to add to the pixel number
# in order to get all the pixels in a sizeXsize grid

def get_neighbors(size):
	neighbors = []
	i = 0
	j = 0
	while i < size:
		while j < size:
			neighbors.append(28*i+j)
			j += 1
		i += 1
	return neighbors

# Function that returns a list of new pixel values based on restructing
# the 28x28 pixel grid (provided as row) as a 28/size x 28/size grid.
# size must be a factor of 28

def restructure(row, size):
	restructured = []
	neighbors = get_neighbors(size)
	for i in range(28):
		if i%size==0:
			for j in range(28):
				if j%size==0:
					total = 0
					for k in neighbors:
						total += row[28*i + j + k]
					restructured.append(float(total)/(size*size))
	return restructured

# Read in the training data: the pixel-value data on 28x28 grid
# for 42,000 data points.
# Column "label" contains the correct number.
# Column "pixelx" contains the darkness of pixel x
# where x = 28*i + j (i is row and y is column)
# x is between 0 and 783, inclusive.
# Pixel-value is between 0 and 255, inclusive.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Create a list of the feature column names
collist = train_data.columns.tolist()
features_col = collist[1:]

# Select the training and test data
features_train = train_data[features_col]
labels_train = train_data['label']

features_test = test_data[features_col]

gridsize = 2    # Must be a factor of 28

# Restucture the training data, and save the output to a list of lists (new_data_train)
print("Starting to restructure the training data...")
new_data_train = []
for i in range(42000):
	new_row = restructure(features_train.iloc[i], gridsize)
	new_data_train.append(new_row)
    # These provide feedback about progress
	if i == 10499:
		print("quarter done")
	if i == 20999:
		print("halfway done")
	if i == 31499:
		print("3/4 done")
# Create a dataframe of the condensed training data
new_dataframe_train = pd.DataFrame(new_data_train, columns = features_col[:len(new_data_train[0])])

# Repeat the process for the testing data
print("Starting to restructure the training data...")
new_data_test = []
for i in range(28000):
	new_row = restructure(features_test.iloc[i], gridsize)
	new_data_test.append(new_row)
	if i == 6999:
		print("quarter done")
	if i == 13999:
		print("halfway done")
	if i == 20999:
		print("3/4 done")
# Create a new dataframe of the test output
new_dataframe_test = pd.DataFrame(new_data_test, columns = features_col[:len(new_data_test[0])])

# Create decision tree, fit, and predict.
clf = skl.DecisionTreeClassifier(criterion = "entropy", min_samples_leaf = 3)

clf.fit(features_train, labels_train)

output = clf.predict(features_test)
# Rowid just contains the values 1-28000
rowid =[]
for i in range(1,28001):
	rowid.append(i)

d = {"ImageId": rowid, "Label": output}

df = pd.DataFrame(d)
df.to_csv('output.csv', index=False)

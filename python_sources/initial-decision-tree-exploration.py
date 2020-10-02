# **Title: Initial Decision Tree Exploration**
# 
# **Author: Alex Caron**
# 
# I'm interested in exploring some of the parameters of the DecisionTreeClassifier from scikit-learn. I will start by visualizing the accuracy of the model based on different paraments.
# Import various libraries
import numpy as np   #linear algebra
import pandas as pd  #data processing
import matplotlib.pyplot as plt #data visualization
import sklearn.tree as tree #machine learning

# Create a function that will run a decision tree model
# with parameters crit (gini or entropy) and min_leaf.

def run_tree(train_data, test_data, crit, min_leaf):
    
    # Create a list of the feature column names
	collist = train_data.columns.tolist()
	features_col = collist[1:]
    
    # Select the training data and test data
	features_train = train_data[features_col]
	labels_train = train_data['label']

	features_test = test_data[features_col]
	labels_test = test_data['label'].tolist()
    
    # Create, fit, and predict using DecisionTreeClassifier
	clf = tree.DecisionTreeClassifier(criterion = crit, min_samples_leaf = min_leaf)
	clf.fit(features_train, labels_train)
	output = clf.predict(features_test)
    
    # Calculate and return the accuracy
	total = len(output)
	total_corr = 0

	for i in range(total):
		if int(output[i]) == labels_test[i]:
			total_corr += 1
	return float(total_corr)/total
# Read in the training data: the pixel-value data on 28x28 grid
# for 42,000 data points.
# Column "label" contains the correct number.
# Column "pixelx" contains the darkness of pixel x
# where x = 28*i + j (i is row and y is column)
# x is between 0 and 783, inclusive.
# Pixel-value is between 0 and 255, inclusive.
data = pd.read_csv('../input/train.csv')
# Create two sets of train/test data
# Train/test1 are just the first and second halves of the data set
length = data.shape[0]
train_data1 = data[:int(length/2)]   # Select first half of rows (from start : half)
test_data1 = data[int(length/2):]    # Select 2nd half of rows (from half : end)

# Train/test2 have alternate rows from the complete data set
train_data2 = data[::2]              # Select every other row
test_data2 = data[1:][::2]           # Starting with second row, select every other row
# Run a test to determine a good value for min_samples_leaf

# Lists to hold min_samples_leaf value (x)
# and accuracy values (y1, y2) for plotting
x = []
y1 = []
y2 = []

# Run the tree for min_sample_leaf values between 1 and 10
for i in range(1,10):
    acc1 = run_tree(test_data1, train_data1, "entropy", i)
    acc2 = run_tree(test_data2, train_data2, "entropy", i)
    # Add test's results to the lists for plotting
    x.append(i)
    y1.append(acc1)
    y2.append(acc2)

# Create scatterplot of the data
plt.scatter(x, y1, c="blue")
plt.scatter(x, y2, c="orange")
plt.show()
# It looks like the minimum samples in a leaf is fine in the range 1-4 and begins to show decreasing accuracy and increasing variance in the 5-9 range. I will now compare the gini and entropy criterion for the DecisionTreeClassifier.
# Run a test to determine whether gini or entropy works better

# Lists to hold min_samples_leaf value (x)
# and accuracy values (y1, y2) for plotting
x = []
ygini = []
yentr = []

# Run the tree for min_sample_leaf values between 1 and 10
for i in range(1,10):
    accgini = run_tree(test_data2, train_data2, "gini", i)
    # Add test's results to the lists for comparison
    ygini.append(accgini)
    
    accentr = run_tree(test_data2, train_data2, "entropy", i)
    # Add test's results to the lists for comparison
    yentr.append(accentr)

for i in range(len(ygini)):
    diff = ygini[i] - yentr[i]
    if diff > 0:
        print("The gini is more accurate by " + str(diff))
    elif diff < 0:
        print("The entropy is more accurate by " + str(-diff))
    else:
        print("They are the same.")



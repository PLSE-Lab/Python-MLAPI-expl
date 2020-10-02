import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import csv
#%matplotlib inline

def compose_image(data_slice):
    ## Function takes a n x 28 data slice and outputs each row as
    ## a colourmap.

    # Find number of images to display
    nrows = len(data_slice.index)
    nrows_root = math.ceil(math.sqrt(nrows))
    
    # For each image, reshape the data and add to a subplot
    fig = plt.figure(figsize=(4,4))
    for i in range(0,nrows):
        pixel_array = data_slice.iloc[i]
        pixel_array = pixel_array.reshape((28,28)).astype(np.uint8)
        ax = fig.add_subplot(nrows_root,nrows_root,i+1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', 
                        top='off',
                        bottom='off',
                        left='off',
                        right='off'
                      )
        ax.set_aspect('equal')
        ax.imshow(pixel_array, cmap=cm.binary)
    
    # Remove padding
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return

def export_to_csv(y_test):
    #np.savetxt('digit_recognizer.csv', column_contents)
    f = open('digit_recognizer.csv','w')
    result_str = 'ImageId,Label\n'
    total = 1
    for i in range(len(y_test)):
        label = str(y_test[i].astype('int'))
        result_str += str(total) + ',' + label + '\n'
        total += 1
    f.write(result_str)
    return

# Load datasets
data_all = pd.read_csv('../input/train.csv')
data_labels = data_all[[0]]
data_image = data_all.drop(data_all.columns[[0]], axis=1)
x_test = pd.read_csv('../input/test.csv')

# Select a random permutation of images to display
row_selection = np.random.choice(data_image.index, 25)
data_slice = data_image.iloc[row_selection]
compose_image(data_slice)
    
# Split the data into train and test sets
#x_train, x_test, y_train, y_test = train_test_split(
#    data_image, data_labels, test_size = 0.33, random_state = 1)
x_train = data_image
y_train = data_labels

# Train SVM - basic, not optimised
clf = OneVsRestClassifier(LinearSVC(random_state=0, verbose=2))
clf.fit(x_train, y_train)
y_test = clf.predict(x_test)

export_to_csv(y_test)
print('Complete!')

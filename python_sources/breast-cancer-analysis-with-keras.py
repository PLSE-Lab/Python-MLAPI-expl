#!/usr/bin/env python
# coding: utf-8

# ## Analysis of Breast Cancer Tumors
# 
# This Kernel is an exploration and analysis of data taken from breast cancer tumors in patients. The dataset originates from 
# the following link.
# 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# The dataset in question comes from biopsies of breast cancer tumors. A biopsy was performed on each patient using the techinque of Fine Needle Aspiration (FNA), which produces a sample that might look like the following picture [credit: <a href="https://en.wikipedia.org/wiki/Fine-needle_aspiration#/media/File:Pancreas_FNA;_adenocarcinoma_vs._normal_ductal_epithelium_(400x)_(322383635).jpg" >Wikipedia</a>]
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/6/65/Pancreas_FNA%3B_adenocarcinoma_vs._normal_ductal_epithelium_%28400x%29_%28322383635%29.jpg" 
# alt="FNA sample from a pancreatic tumor" style="width:400px;">
# 
# In this picture, the cluster of cells on the left is the tumor, while the cells on the right are normal. From this sample, one measures the features of _each cell_, such as radius, area, symmetry, and so on. Then, the mean, standard deviation, and largest value are reported for this sample. So, when reading the dataset, keep in mind that each row is reporting statistics about the cancer cells from a sample like this image.

# In[ ]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras as keras
import tensorflow as tf

np.random.seed(5)
tf.set_random_seed(5)

data_file = '../input/data.csv'
df = pd.read_csv(data_file)
df.head(3)


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# Seems like this dataset is mostly ready to analyze. The `Unnamed` column seems to be full of `NaN`, so I will drop it. (The UCI webpage where the data originates from also does not seem to acknowledge this column. All the more reason to drop it.)

# In[ ]:


df = df.drop(['Unnamed: 32'],axis=1);


# While we're here, let's turn the diagnosis variable into 0 for benign and 1 for malignant.

# In[ ]:


# Assign 1 for 'M' and 0 for 'B'
df['diagnosis'] = df['diagnosis'].map(lambda s: 1 if s=='M' else 0).astype(int)


# # EDA

# ## Feature Breakdown
# 
# The columns in the data set are not all independent. Each feature (except for ID and Diagnosis) all have
# data columns for their mean, standard deviation, and "worst" (meaning largest, top 3 averaged) values. 
# 
# 1. Id: Patient ID.
# 2. Diagnosis: Either M (malignant) or B (benign).
# 
# The next features are measured from the tumors.
# 
# 3. Radius, measured by averaging distances from the center of a cell to its perimeter.
# 4. Texture, measured by the standard deviation of grey-scale values.
# 5. Perimeter.
# 6. Area.
# 7. Smoothness, measured by local deviation in radius length.
# 8. Compactness, defined as $\frac{P^2}{A} - 1$, where $P$ is the perimeter of the cell and $A$ is the area.
# 9. Concavity, which is the worst concave portion of the cell. 
# 10. Concavity Points, which is the number of concave portions in the perimeter of the cell.
# 11. Fractal Dimension. I was not able to find precise definitions of how they caluclate this, but one way this could have been done is by calculating the ratios of $-\log(N)/\log(d)$, where the cancer cell's perimeter is approximated by straight lines of length $d$ that total to a distance $N$. More on this can be found <a href="http://fractalfoundation.org/OFC/OFC-10-4.html">here</a>.

# In[ ]:


feature_columns = list(df.columns)[2:]

# The following line is just some string-foo to split off the main feature name from the mean, se, and worst.
general_features = ['_'.join(s.split('_')[:-1]) for s in feature_columns[0:10]]

print(general_features)


# ## Radius, Area, Perimeter

# One thing to notice: `radius`, `area`, and `perimeter` are highly dependent, which is expected. If each cell is roughly a disk, then $A = \pi r^2$, while $P = 2 \pi r$.

# In[ ]:


sns.pairplot(df[['radius_mean','area_mean','perimeter_mean']])


# As expected, both area and perimeter follow the patterns for disks: namely $ A = \pi r^2$ and $P = 2 \pi r$. Indeed, in the `area` vs. `radius` plot, a radius of $20$ should give an area of about $1200$, which is consistent with this plot. In the `perimeter` vs. `radius` plot, the points around a radius of 10 should have a perimeter near $60$, which is also consistent. 
# 
# Now, this does not allow us to _conclude_ that the cells are ciruclar. After all, the radius of _each cell_ is an average of radii, and this is then further averaged over the entire person's biopsy sample. So, one data point in a feature column ending in `_mean` is the result of two different kinds of averages. The initial averaging over different radii is almost like finding a circle that approximates the cell's shape well. 
# 
# Despite the multiple averages occuring, it is nice to see that our data reflects the circular nature of these cells.
# 

# ## Inner-feature comparisons

# The same type of feature occurs every 10 columns as either mean, standard error, or worst. So, we make a function that plucks these out for us to analyze.

# In[ ]:


# Feature should be a string.
def select_feature(df, feature):
    related_features = [feature+'_'+suffix for suffix in ['mean','se','worst']]
    return df[related_features]

select_feature(df, 'radius').head()


# Let's compare the statistics between some of these features:

# In[ ]:


sns.pairplot(select_feature(df, 'radius'))
sns.pairplot(select_feature(df, 'texture'))
sns.pairplot(select_feature(df, 'fractal_dimension'))


# # Deep Network with Keras

# ## Model Description
# 
# Let's try out a deep network with one input layer, one output layer that has one unit to measure the probability of malignancy, and one hidden unit.
# In what follows, I will run cross-validation, where in each instance I will fit a `MinMaxScaler` to the training set and then apply this scaler to the test set.
# 
# This model currently ignores the `_se` and `_worst` features for the sake of consistency; after all, since the same feature shows up with `_mean`, `_se`, and `_worst`, 
# it would be strange to scale these three components of a feature independently. Perhaps in a later version of this kernel I will incorporate this.

# In[ ]:


def run_cross_val(X, y, num_folds, num_input_units, num_hidden_units):
    kfold = StratifiedKFold(n_splits=num_folds)
    cross_scores = []
    histories = []
    for train, test in kfold.split(X, y):
        scaler = MinMaxScaler(feature_range=(0,1))

        X_train = scaler.fit_transform(X[train])
        X_test = scaler.transform(X[test])
        y_train = y[train]
        y_test = y[test]

        dnn_model = keras.models.Sequential()

        # Input layer
        dnn_model.add(
            keras.layers.Dense(
                units= num_input_units,
                input_dim = X.shape[1],
                kernel_initializer = 'glorot_uniform',
                bias_initializer = 'zeros',
                activation = 'tanh'
            ))

        # hidden layer
        dnn_model.add(
            keras.layers.Dense(
                units=num_hidden_units,
                input_dim = num_input_units,
                kernel_initializer = 'glorot_uniform',
                bias_initializer = 'zeros',
                activation = 'tanh'
            ))

        #output layer
        dnn_model.add(
            keras.layers.Dense(
                units = 1,
                input_dim = num_hidden_units,
                kernel_initializer = 'glorot_uniform',
                bias_initializer = 'zeros',
                activation = 'sigmoid'
            ))

        sgd_optimizer = keras.optimizers.SGD(lr = 0.0001, decay = 1e-7, momentum = .9)

        dnn_model.compile(optimizer = sgd_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        history = dnn_model.fit(X_train, y_train, epochs = 400, validation_split=0.33, batch_size = 15, verbose = 0)
        histories.append(history.history) # save the histories for later use

        scores = dnn_model.evaluate(X_test, y_test, verbose = 0)

        print("%s: %.2f%%" % (dnn_model.metrics_names[1], scores[1]*100))

        cross_scores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cross_scores), np.std(cross_scores)))
    
    return histories, cross_scores
    
    


# In[ ]:


num_input_units = 20
num_hidden_units = 20

X = df.drop(['id','diagnosis'], axis=1).values[:, 0:10]
y = df['diagnosis'].values

print('Full Accuracy Scores:\n')
full_histories, full_cross_scores = run_cross_val(X, y, 10, num_input_units, num_hidden_units)

X = df.drop(['id','diagnosis','area_mean','perimeter_mean'], axis=1).values[:, 0:8]

print('Ablated Accuracy Scores:\n')
ablated_histories, ablated_cross_scores = run_cross_val(X, y, 10, num_input_units, num_hidden_units)


# The following plots are made from running a second cross-validation (where the folding is done on each `X_train` instead of the whole dataset).

# In[ ]:


fold_idx = 0
fig, ax = plt.subplots(2, 2, figsize = (10, 10))
results = [full_histories, ablated_histories]
keywords = ['loss','acc']

for i in range(2):
    for j in range(2):            
            ax[i,j].plot(results[i][fold_idx][keywords[j]])
            ax[i,j].plot(results[i][fold_idx]['val_'+keywords[j]])
            ax[i,j].set_xlabel('Epochs')
            ax[i,j].set_ylabel(keywords[j])
            ax[i,j].legend(['Fold Train', 'Fold Test'], loc = 'lower left')
            if i==0:
                ax[i,j].set_title('Non-Ablated')
            else:
                ax[i,j].set_title('Ablated')
            if j==1:
                ax[i,j].set_ylim((0,1))
                


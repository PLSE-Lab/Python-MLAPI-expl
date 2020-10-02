#!/usr/bin/env python
# coding: utf-8

# # Benign or Malignant?
# 
# In this notebook, I will attempt to predict whether cancer is benign or malignant based on features that are computed from a digitized image of a fine needle aspirate of a breast mass. Data is from the Breast Cancer Wisconsin data set. I wil aslo use seaborn to visualize the data and help select the best features for my model. 
# 
# **Outline:**
# 1.  Imports
# 2. Load/Preview data using Pandas
# 3. Explore The Dependant Variable (diagnosis)
# 4. Explore The Independant/Predictor Variables and Select Features
# 5. Machine Learning To Predict Diagnosis (XGBoost Classifier)
# 

# # Imports:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data Handling:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Other:
import warnings
warnings.filterwarnings('ignore')

# Machine Learning:
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Import/Preview Data:

# In[ ]:


# First load the CSV to a Pandas dataframe
data_DF = pd.read_csv('../input/data.csv')

# Next examine the Dataframe
print(data_DF.shape)
print(data_DF.keys())
print(data_DF.dtypes)


# In[ ]:


# view the top 5 rows:
data_DF.head()


# In[ ]:


# Look at some basic stats of all numeric data:
data_DF.describe()


# Here we quickly check for missing values. We can see that the variable named 'Unnamed: 32' is missing for everyone... We will go ahead and drop that variable from the dataframe right away. 

# In[ ]:


data_DF.isnull().sum()


# In[ ]:


data_DF = data_DF.drop(labels = ['Unnamed: 32'], axis = 1)


# # Let's Quickly Explore The Dependant Variable (diagnosis)

# In[ ]:


sns.countplot(data_DF.diagnosis);


# Here we can see that there are ~50% more Benign ('B') diagnoses than Malignant ('M'). Let's see what those actual numbers are, though...

# In[ ]:


data_DF.diagnosis.value_counts()


# # Explore The Independant/Predictor Variables 

# First, I'll just quickly look at how each variable differs between benign an malignant diagnoses. I prefer to do this with boxplots to help vizualize the spread/outliers of each variable, but this could also be done with violin plots or other options.

# **How each variable differs by diagnosis:**

# In[ ]:


vars = data_DF.keys().drop(['id','diagnosis'])
plot_cols = 5
plot_rows = math.ceil(len(vars)/plot_cols)

plt.figure(figsize = (5*plot_cols,5*plot_rows))

for idx, var in enumerate(vars):
    plt.subplot(plot_rows, plot_cols, idx+1)
    sns.boxplot(x = 'diagnosis', y = var, data = data_DF)


# Alright, That's a lot of graphs. We can see that for a lot of the predictor variables, values are higher (on average) in the malignant group. However, there are plenty of outliers amongst the benign data, which could cause those values to look like those of the malignant group... next, it is probably interesting to see if/how any of these variables are correlated with eachother...

# **Correlations Heat Map:**

# In[ ]:


fig, (ax) = plt.subplots(1, 1, figsize=(20,10))

hm = sns.heatmap(data_DF.corr(), 
                 ax=ax, # Axes in which to draw the plot
                 cmap="coolwarm", # color-scheme
                 annot=True, 
                 fmt='.2f',       # formatting  to use when adding annotations.
                 linewidths=.05)

fig.suptitle('Breast Cancer Correlations Heatmap', 
              fontsize=14, 
              fontweight='bold');


# From this correlation heat map, we have a bit of an idea which variables may be a little redundant, and could be removed because of too much covariance...
# 
# * radius_mean, perimeter_mean, area_mean are all highly correlated (r= 0.99 - 1.00). This is intuitive. 
# * The above 3 vars are also highly correlated with radius_worst, perimeter_worst, area_worst...
# 
# maybe I'll keep the 'area_worst' or 'area_mean' and drop the rest...
# 
# * We can also see that compactness_worst, concave_points_worst, concavity_worst also are highly correlated with compactness_mean, concave_mean, and Concavity_mean
#     - concave_points_mean and concave_points_worst both seems to have distinct differences between benign and malignant, with relatively lower outliers. Maybe we should choose one of these to keep and drop the rest?
# * There are a few more vars with correlations > 0.80. We'll remove some of those too...    

# In[ ]:


vars_to_drop = ['id', 'radius_mean', 'perimeter_mean', 'radius_worst', 'area_worst', 'perimeter_worst', 'radius_se', 'perimeter_se',
               'concave points_mean', 'compactness_mean', 'compactness_worst', 'concavity_worst', 'concavity_mean', 'concavity_se',
               'texture_worst', 'smoothness_worst', 'texture_se']


# **Now that we've dropped some variables, we can further visualize it... **
# 
# (this is admittedly too many variables for this type of visualization, but if you have a decent sized monitor, it's still doable)

# In[ ]:


g = sns.pairplot(data_DF.drop(vars_to_drop, axis = 1), hue='diagnosis', height=3)
g.map_lower(sns.kdeplot)


# # Machine Learning to Predict the Diagnosis:

# Random Forest Classifier models tend to do really well with these types of predictions (and are also less sensitive to covariance in the predictor variables, so we will start there. I will use a simple XGBoost classifier model to perform these predictions.

# **First, Define X, y**

# In[ ]:


y = data_DF.diagnosis
X = data_DF.drop(vars_to_drop, axis = 1)
X = X.drop('diagnosis', axis = 1)
print(y.shape)
print(X.shape)
X.head()


# **Next, Split into Training and validation data:**

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size = 0.25)


# **Create Simple XGBoost Classifier Model to Predict Diagnosis:**

# In[ ]:


XGBC_model = XGBClassifier(random_state=1, objective = 'multi:softprob', num_class=2)
XGBC_model.fit(train_X, train_y)

# make predictions
XGBC_predictions = XGBC_model.predict(val_X)

# Print Accuracy for initial RF model
XGBC_accuracy = accuracy_score(val_y, XGBC_predictions)
print("Accuracy score for XGBoost Classifier model : " + str(XGBC_accuracy))


# Without any tuning, we are able to predict if the biopsied tumor is benign or malignant with ~97% accuracy using an XGBoosted Random Forest Classifier Model... Not too Bad...
# 
# **Next, let's examine the importance of the predictive variables**

# In[ ]:


# Create dataframe of feature name and importance 
feature_imp_DF = pd.DataFrame({'feature': X.keys().tolist(), 'importance': XGBC_model.feature_importances_})

# Print the sorted values form the dataframe:
print("Feature Importance:\n")
print(feature_imp_DF.sort_values(by=['importance'], ascending=False))


# In[ ]:


# Plot feature importance from the dataframe..


# In[ ]:


# feature_imp_DF.sort_values(by=['importance'],ascending=False).plot(kind='bar');
g = sns.barplot(x="feature", y="importance", data=feature_imp_DF.sort_values(by=['importance'],ascending=False))
g.set_xticklabels(feature_imp_DF.sort_values(by=['importance'],ascending=False)['feature'],rotation=90);


# In[ ]:





# In[ ]:





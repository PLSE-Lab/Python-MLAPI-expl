#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotting libs
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#sklearn libs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First we have to load the dataset.

# In[ ]:


star_data = pd.read_csv("../input/star-dataset/6 class csv.csv")
star_data.head()


# We have 2 categorical features "Star color" and "Spectral Class". A google search reveals that both couls be useful since every spectral class is associated with a certain color. The star type in this dataset is the feature that we want to predict. Additionally we create a dictionary with the corrsponding types and add it to this dataset. The only reason we are doing this is because it will be useful when we visualize the data.

# In[ ]:


star_types = {0:"Brown Dwarf", 1:"Red Dwarf", 2:"White Dwarf", 3:"Main Sequence", 4:"Supergiant", 5:"Hypergiant"}
star_data["Star type Decoded"] = star_data["Star type"].map(star_types) 


# Next we need to check if there is missing data.

# In[ ]:


star_data.isnull().sum()


# As we can see there is no missing data. Next we should take a look at the categorical features and on how we want to deal with them. If there are entries with the same meaning that only differ in the way they are written then we should replace them.

# In[ ]:


star_data["Spectral Class"].value_counts()


# As we can see we have 7 different Spectral classes which we could encode right now. But first let's check the star color feature.

# In[ ]:


star_data["Star color"].value_counts()


# As mentioned above there are colors that are written just a bit differently.   

# In[ ]:


star_data["Star color"] = star_data["Star color"].str.lower()
star_data["Star color"] = star_data["Star color"].str.replace(' ','')
star_data["Star color"] = star_data["Star color"].str.replace('-','')
star_data["Star color"] = star_data["Star color"].str.replace('yellowwhite','whiteyellow')
star_data["Star color"].value_counts()


# This is much better and now we can encode every label. Here I will use a LabelEncoder.

# In[ ]:


le_specClass = LabelEncoder()
star_data["SpecClassEnc"] = le_specClass.fit_transform(star_data["Spectral Class"])
print("Encoded Spectral Classes: " + str(le_specClass.classes_))


# We do the same thing for the star colors.

# In[ ]:


le_starCol = LabelEncoder()
star_data["StarColEnc"] = le_starCol.fit_transform(star_data["Star color"])
print("Encoded Star colors: " + str(le_starCol.classes_))


# Since this is a small dataset with just a few features we can use a pairplot to visualize all combinations between these features.

# In[ ]:


sns.pairplot(star_data.drop(["Star color", "Spectral Class"], axis=1), hue="Star type Decoded", diag_kind=None)
plt.show()


# From this we see that one of our most important features we will be Absolute magnitude(Mv) since we already see that in combination with every other feature every star type can be easily separated by each other. If we plot the "Absolute magnitude(Mv)" over the "Spectral Class" with regard to the different "Star types" then we can see the separation of each type more clearly. (note: if you google for spectral classes you will find the same picture)

# In[ ]:


sns.catplot(x="Spectral Class", y="Absolute magnitude(Mv)", data=star_data, hue="Star type Decoded", order=['O', 'B', 'A', 'F', 'G', 'K', 'M'], height=9)
plt.gca().invert_yaxis()


# Before we create our model we should split it into a training and a test set. Then use the StandardScaler() on both of them. 

# In[ ]:


#select all features for learning and the feature we want to predict
x = star_data.select_dtypes(exclude="object").drop("Star type", axis=1)
y = star_data["Star type"]

#split out dataset into a training and a test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=star_data["Star type"])

scaler = StandardScaler()

#use the scaler to scale our data
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)

#since we want to have our dataframe we need to replace it in the corresponding sets
x_train = pd.DataFrame(x_train_sc, index=x_train.index, columns=x_train.columns)
x_test = pd.DataFrame(x_test_sc, index=x_test.index, columns=x_test.columns)


# Since we have 6 different star types and we want to train on every class we use stratify from train_test_split to use 80% of all features for every type in our training dataset. Also we use the standardscaler separately on our training and our test set. This way we can prevent leaking of the information about the distribution of our test set into our model.
# 
# Finally we can start building our model. Here we will use the XGBoost classifier.

# In[ ]:


xgb = XGBClassifier(n_estimators=1000, n_jobs=-1, random_state=42)

xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


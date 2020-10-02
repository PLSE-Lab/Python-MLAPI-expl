#!/usr/bin/env python
# coding: utf-8

# # Forest Cover Type Prediction

# *Forest cover in general refers to the relative or sure land area that is covered by forests or the forest canopy or open woodland.* --Wikipedia <br> <br>
# **In this model we will try to predict the forest type for a given forest based**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report


# In[ ]:


dataset = pd.read_csv('../input/covtype.csv')


# ## Analyzing the dataset

# Here, we have 581012 rows and 55 columns in this dataset

# In[ ]:


dataset.shape


# In[ ]:


dataset.head()


# In the dataset, we have the following variables:
# > 1. Elevation : Elevation in meters
# > 2. Aspect : Aspect in degrees azimuth
# > 3. Slope : Slope in degrees
# > 4. Horizontal_Distance_To_Hydrology : Horizontal distance to nearet surface water features
# > 5. Vertical_Distance_To_Hydrology : Vertical distance to nearet surface water features
# > 6. Horizontal_Distance_To_Roadways : Horizontal distance to nearest roadway
# > 7. Hillshade_9am : Hillshade index at 9am
# > 8. Hillshade_Noon : Hillshade index at noon
# > 9. Hillshade_3pm : Hillshade index at 3pm
# > 10. Horizontal_Distance_To_Fire_Points : Horizontal dist to nearest wildfire ignition points
# > 11. Widerness_Area : 4 binary columns (0->absence, 1->presence)
# > 12. Soil_Type : 40 binary columns (0->absence, 1->presence)
# > 13. Cover_Type : integet (1 to 7)

# The first ten columns have numerical value, and the rest columns, namely, Wilderness_Area and Soil_Type, these are categorical variables with 4 and 40 different categories respectively.

# ### Let's first analyze our categorical variables.

# #### Analyzing soil type

# In[ ]:


#plot 1
columns = ["Soil_Type"+str(i) for i in range(1,41)]
count_ones = []
for i in columns:
    count_ones.append(dataset[dataset[i]==1][i].count())
y_pos = np.arange(len(columns))
plt.figure(figsize=(10,5))
plt.bar(y_pos, count_ones, align="center", alpha=0.5)
plt.xticks(y_pos, [i for i in range(1,41)])
plt.ylabel("Number of Positive Examples")
plt.xlabel("Soil Type")
plt.title("Soil Type Analysis")
plt.show()

#plot 2
columns = ["Soil_Type"+str(i) for i in range(1,41)]
count_zeros = []
for i in columns:
    count_zeros.append(dataset[dataset[i]==0][i].count())
y_pos = np.arange(len(columns))
plt.figure(figsize=(10,5))
plt.bar(y_pos, count_zeros, align="center", alpha=0.5)
plt.xticks(y_pos, [i for i in range(1,41)])
plt.ylabel("Number of Negative Examples")
plt.xlabel("Soil Type")
plt.title("Soil Type Analysis")
plt.show()


# Here, we may clearly see that our data is imbalanced. Soil_Type29 is prevalent among all the forests. Though, there are some soil types such as 7, 36 etc. which are very rare. So, there could be some direct relations between these rare soil types and their respective Cover Type. Let's find out.

# In[ ]:


dataset[dataset['Soil_Type7']==1]['Cover_Type'].value_counts()


# As expected, the presence of soil_type7 always has its corresponding Cover_Type as 2

# #### Analyzing the cover type

# In[ ]:


dataset['Cover_Type'].value_counts()


# ## Preprocessing the dataset

# The dataset is already preprocessed. There are no Null values. Also, all the categorical variables are also converted to dummy variables.

# In[ ]:


y = dataset['Cover_Type']
x = dataset.drop(['Cover_Type'], axis=1)


# In[ ]:


x = scale(x)


# #### Splitting the dataset

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


len(x_train), len(x_test)


# #### Fitting the model

# In[ ]:


lr_clf = LogisticRegression(penalty='l1', C=0.1)
lr_clf.fit(x_train, y_train)


# In[ ]:


predictions = lr_clf.predict(x_test)
print(classification_report(predictions, y_test))


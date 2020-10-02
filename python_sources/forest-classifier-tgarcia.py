#!/usr/bin/env python
# coding: utf-8

# # Classify forest types based on information about the area
# predict what types of trees there are in an area based on various geographic features.
# The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:
# 
# 1 - Spruce/Fir
# 2 - Lodgepole Pine
# 3 - Ponderosa Pine
# 4 - Cottonwood/Willow
# 5 - Aspen
# 6 - Douglas-fir
# 7 - Krummholz
# 
# The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).
# Data Fields
# Elevation - Elevation in meters
# Aspect - Aspect in degrees azimuth
# Slope - Slope in degrees
# Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation
# 
# The wilderness areas are:
# 
# 1 - Rawah Wilderness Area
# 2 - Neota Wilderness Area
# 3 - Comanche Peak Wilderness Area
# 4 - Cache la Poudre Wilderness Area
# 
# The soil types are:
# 
# 1 Cathedral family - Rock outcrop complex, extremely stony.
# 2 Vanet - Ratake families complex, very stony.
# 3 Haploborolis - Rock outcrop complex, rubbly.
# 4 Ratake family - Rock outcrop complex, rubbly.
# 5 Vanet family - Rock outcrop complex complex, rubbly.
# 6 Vanet - Wetmore families - Rock outcrop complex, stony.
# 7 Gothic family.
# 8 Supervisor - Limber families complex.
# 9 Troutville family, very stony.
# 10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
# 11 Bullwark - Catamount families - Rock land complex, rubbly.
# 12 Legault family - Rock land complex, stony.
# 13 Catamount family - Rock land - Bullwark family complex, rubbly.
# 14 Pachic Argiborolis - Aquolis complex.
# 15 unspecified in the USFS Soil and ELU Survey.
# 16 Cryaquolis - Cryoborolis complex.
# 17 Gateview family - Cryaquolis complex.
# 18 Rogert family, very stony.
# 19 Typic Cryaquolis - Borohemists complex.
# 20 Typic Cryaquepts - Typic Cryaquolls complex.
# 21 Typic Cryaquolls - Leighcan family, till substratum complex.
# 22 Leighcan family, till substratum, extremely bouldery.
# 23 Leighcan family, till substratum - Typic Cryaquolls complex.
# 24 Leighcan family, extremely stony.
# 25 Leighcan family, warm, extremely stony.
# 26 Granile - Catamount families complex, very stony.
# 27 Leighcan family, warm - Rock outcrop complex, extremely stony.
# 28 Leighcan family - Rock outcrop complex, extremely stony.
# 29 Como - Legault families complex, extremely stony.
# 30 Como family - Rock land - Legault family complex, extremely stony.
# 31 Leighcan - Catamount families complex, extremely stony.
# 32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
# 33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
# 34 Cryorthents - Rock land complex, extremely stony.
# 35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
# 36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
# 37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
# 38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
# 39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
# 40 Moran family - Cryorthents - Rock land complex, extremely stony.

# In[ ]:


# Import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Input data files 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import input data
train = pd.read_csv("../input/learn-together/train.csv")
test = pd.read_csv("../input/learn-together/test.csv")


# ## Exploratory Data Analysis

# In[ ]:


train.head()


# In[ ]:


train.describe()


# ### Train and Test Dimensions
# Test data set has more elements than the training data set

# In[ ]:


print("Training Dimensions: %s" % str(train.shape)) 
print("Test Dimensions: %s" % str(test.shape)) 


# In[ ]:


# Drop ID in the training data
# Source: https://www.kaggle.com/kayveen/simple-notebook-to-make-a-first-submission

train = train.drop(["Id"], axis = 1)
test_ids = test["Id"]
test = test.drop(["Id"], axis = 1)
X_Test = test

# split data into training (80%) and validation data (20%)
X_Train, X_Val, Y_Train, Y_Val = train_test_split(train.drop(['Cover_Type'],axis=1), train['Cover_Type'], test_size=0.2, random_state=42)


# ## Model

# ### 1.  Random Forest
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)

# In[ ]:


model = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=42)
model.fit(X_Train, Y_Train)


# In[ ]:


# Model Score
model.score(X_Train, Y_Train)


# In[ ]:


# Prediction
predictions = model.predict(X_Val)
accuracy_score(Y_Val, predictions)


# In[ ]:


# Predictions in Test Set
Y_Test = model.predict(X_Test)


# In[ ]:


# Submision File
# Save test predictions to file
output = pd.DataFrame({'ID': test_ids,
                       'Cover_Type': Y_Test})
output.to_csv('submission.csv', index=False)


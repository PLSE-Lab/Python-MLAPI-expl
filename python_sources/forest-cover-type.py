#!/usr/bin/env python
# coding: utf-8

# # [Forest Cover Type Dataset](https://archive.ics.uci.edu/ml/datasets/Covertype)
# **Tree types found in the Roosevelt National Forest in Colorado**

# ## Context
# This dataset contains tree observations from four areas of the Roosevelt National Forest in Colorado. All observations are cartographic variables (no remote sensing) from 30 meter x 30 meter sections of forest. There are over half a million measurements total!

# ## Data Set Information:
# 
# Predicting forest cover type from cartographic variables only (no remotely sensed data). The actual forest cover type for a given observation (30 x 30 meter cell) was determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. Independent variables were derived from data originally obtained from US Geological Survey (USGS) and USFS data. Data is in raw form (not scaled) and contains binary (0 or 1) columns of data for qualitative independent variables (wilderness areas and soil types). 
# 
# This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices. 
# 
# Some background information for these four wilderness areas: Neota (area 2) probably has the highest mean elevational value of the 4 wilderness areas. Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational value, while Cache la Poudre (area 4) would have the lowest mean elevational value. 
# 
# As for primary major tree species in these areas, Neota would have spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5). Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6), and cottonwood/willow (type 4). 
# 
# The Rawah and Comanche Peak areas would tend to be more typical of the overall dataset than either the Neota or Cache la Poudre, due to their assortment of tree species and range of predictive variable values (elevation, etc.) Cache la Poudre would probably be more unique than the others, due to its relatively low elevation range and species composition.

# ## Attribute Information:
# 
# Given is the attribute name, attribute type, the measurement unit and a brief description. The forest cover type is the classification problem. The order of this listing corresponds to the order of numerals along the rows of the database. 
# 
# In regards to the number of attributes, we have 12 measurements spread out over 54 columns of data. Ten are quantitative variables, four binary are wilderness areas, and 40 binary are soil type variables.
# 
# You can find more information about the data [here](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info).
# 

# ## Acknowledgement
# This dataset is part of the UCI Machine Learning Repository, and the original source can be found [here](https://archive.ics.uci.edu/ml/datasets/Covertype). The original database owners are Jock A. Blackard, Dr. Denis J. Dean, and Dr. Charles W. Anderson of the Remote Sensing and GIS Program at Colorado State University.

# ---

# # Import Necessary Libraries

# In[ ]:


import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


# ---

# # Read Data

# In[ ]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/covtype.csv')
df.head()


# In[ ]:


print('Features: {} \nObservations: {}'.format(df.shape[1], df.shape[0]))


# In[ ]:


print(df.columns)


# All columns are proberbly named

# ## Finding
# 
# * The dataset has 54 features and 1 target variable `Cover_Type`.
# * From the 54 features, 10 are numerical and 44 are categorical.
# * From the categorical data 4 are of `Wilderness_Area` and 40 are of `Soil_Type`
# 
# ### Categorical Data
# With the informations from [UCI](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info) we know the correct names of all `Soil_Types` and `Wilderness_Areas`. See table below.
# 
# We will keep the current feature naming, but might take a closer look later if necessary.

# Feature Name    |    Names
# ---------------|:--------
# Wilderness_Area1    |    Rawah Wilderness Area
# Wilderness_Area2    |    Neota Wilderness Area
# Wilderness_Area3    |    Comanche Wilderness Area
# Wilderness_Area4    |    Cache La Poudre Wilderness Area
# Soil_Type1    |    Cathedral family - Rock outcrop complex, extremely stony
# Soil_Type2    |    Vanet - Ratake families complex, very stony
# Soil_Type3    |    Haploborolis - Rock outcrop complex, rubbly
# Soil_Type4    |    Ratake family - Rock outcrop complex, rubbly
# Soil_Type5    |    Vanet family - Rock outcrop complex, rubbly
# Soil_Type6    |    Vanet - Wetmore families - Rock outcrop complex, stony
# Soil_Type7    |    Gothic family
# Soil_Type8    |    Supervisor - Limber families complex
# Soil_Type9    |    Troutville family, very stony
# Soil_Type10    |    Bullwark - Catamount families - Rock outcrop complex, rubbly
# Soil_Type11    |    Bullwark - Catamount families - Rock land complex, rubbly
# Soil_Type12    |    Legault family - Rock land complex, stony
# Soil_Type13    |    Catamount family - Rock land - Bullwark family complex, rubbly
# Soil_Type14    |    Pachic Argiborolis - Aquolis complex
# Soil_Type15    |    unspecified in the USFS Soil and ELU Survey
# Soil_Type16    |    Cryaquolis - Cryoborolis complex
# Soil_Type17    |    Gateview family - Cryaquolis complex
# Soil_Type18    |    Rogert family, very stony
# Soil_Type19    |    Typic Cryaquolis - Borohemists complex
# Soil_Type20    |    Typic Cryaquepts - Typic Cryaquolls complex
# Soil_Type21    |    Typic Cryaquolls - Leighcan family, till substratum complex
# Soil_Type22    |    Leighcan family, till substratum, extremely bouldery
# Soil_Type23    |    Leighcan family, till substratum, - Typic Cryaquolls complex.
# Soil_Type24    |    Leighcan family, extremely stony
# Soil_Type25    |    Leighcan family, warm, extremely stony
# Soil_Type26    |    Granile - Catamount families complex, very stony
# Soil_Type27    |    Leighcan family, warm - Rock outcrop complex, extremely stony
# Soil_Type28    |    Leighcan family - Rock outcrop complex, extremely stony
# Soil_Type29    |    Como - Legault families complex, extremely stony
# Soil_Type30    |    Como family - Rock land - Legault family complex, extremely stony
# Soil_Type31    |    Leighcan - Catamount families complex, extremely stony
# Soil_Type32    |    Catamount family - Rock outcrop - Leighcan family complex, extremely stony
# Soil_Type33    |    Leighcan - Catamount families - Rock outcrop complex, extremely stony
# Soil_Type34    |    Cryorthents - Rock land complex, extremely stony
# Soil_Type35    |    Cryumbrepts - Rock outcrop - Cryaquepts complex
# Soil_Type36    |    Bross family - Rock land - Cryumbrepts complex, extremely stony
# Soil_Type37    |    Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony
# Soil_Type38    |    Leighcan - Moran families - Cryaquolls complex, extremely stony
# Soil_Type39    |    Moran family - Cryorthents - Leighcan family complex, extremely stony
# Soil_Type40    |    Moran family - Cryorthents - Rock land complex, extremely stony

# **Note:** from that list above there always only one combination available, well possible I should say. For example, `Soil_Type23` has a `1` and `Wilderness_Area1` has a `1`, all other for that observation will have a `0`.

# ### Numerica Data
# 
# As shown below in the table we have different data representation, some are in meters, degrees or a value between 0 and 255 as index.
# 
# Name                                  |   Data Type   | Measurement      | Description
# --------------------------------------|---------------|------------------|------------
# Elevation                             |  quantitative |   meters         |   Elevation in meters
# Aspect                                |  quantitative |   azimuth        |   Aspect in degrees azimuth
# Slope                                 |  quantitative |   degrees        |   Slope in degrees
# Horizontal_Distance_To_Hydrology      |  quantitative |   meters         |   Horz Dist to nearest surface water features
# Vertical_Distance_To_Hydrology        |  quantitative |   meters         |   Vert Dist to nearest surface water features
# Horizontal_Distance_To_Roadways       |  quantitative |   meters         |   Horz Dist to nearest roadway
# Horizontal_Distance_To_Fire_Points    |  quantitative |   meters         |   Horz Dist to nearest wildfire ignition points
# Hillshade_9am                         |  quantitative |   0 to 255 index |   Hillshade index at 9am, summer solstice
# Hillshade_Noon                        |  quantitative |   0 to 255 index |   Hillshade index at noon, summer soltice
# Hillshade_3pm                         |  quantitative |   0 to 255 index |   Hillshade index at 3pm, summer solstice
# 
# We might ask ourself, do we need to do any data conversion here or not. For right now we will leave the data as is. And may come back to it when needed.

# ### Target Variable
# 
# The target variable `Cover_Type` is of type integer and ranges from `1` and `7` and representes a type of tree, eg. Douglas-Fir.
# 
# 
# Key | Class
# ----|--------------
# 1   | Spruce/Fir
# 2   | Lodgepole Pine
# 3   | Ponderosa Pine
# 4   | Cottonwood/Willow
# 5   | Aspen
# 6   | Douglas-fir
# 7   | Krummholz
# 

# ---

# # Data Exploration

# In[ ]:


print(df.info())


# In[ ]:


df.isnull().values.any()


# **Note:** Great! We don't have any null values and of any odd data type.

# In[ ]:


# Handling Duplicates
df.drop_duplicates(keep='first')
df.shape


# **Note:** No Duplicate data found.

# ## Feature Statistics
# 
# We will split the dataset into `numerical` and into `categorical` data. And put the target variable `Cover_Type` into its own df.

# In[ ]:


# Create different datasets by type and area
cont_df = df.loc[:,'Elevation':'Horizontal_Distance_To_Fire_Points']
cat_df  = df.loc[:,'Wilderness_Area1':'Soil_Type40']
wild_df = df.loc[:,'Wilderness_Area1': 'Wilderness_Area4']
soil_df = df.loc[:,'Soil_Type1':'Soil_Type40']
target  = df['Cover_Type']


# ## Continues Data
# 
# We will look at the statistics of numerical features and extract useful informations.

# In[ ]:


# pick number of columns
ncol = 2
# make sure enough subplots
nrow = math.floor((len(cont_df.columns) + ncol - 1) / ncol)
# create the axes
height = 6 * nrow
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))

# go over a linear list of data
for i, col in enumerate(cont_df.columns):
    # compute an appropriate index (1d or 2d)
    ix = np.unravel_index(i, ax.shape) 

    sns.distplot(cont_df[col], ax=ax[ix])

plt.tight_layout()
plt.show();


# In[ ]:


cont_df.describe()


# ### Findings:
# 
# * The `mean` of the features vary from as low as `14` to as high as `2959`. This is due to the different type of measurements, i.e. degrees vs meters.
# * The `standard deviation` (std) tells us how the spread of the data is from the mean. For example, `Horizontal_Distance_To_Hydrology` has a wide spread, but `Hillshade_Noon` has a narrow spread.
# * With the exception of `Elevation` and `Vertical_Distance_To_Hydrology` all other features have a minimum value of `0`.
# * With maybe one exception our data is skewed. 
# * Slope looks off, but for time being we shall ignore it.

# ## Feature Skewness (Continues Data)

# In[ ]:


print(cont_df.skew())


# In[ ]:


skew_df = pd.DataFrame(cont_df.skew(), columns=['Skewness'])

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(16, 8))
sns.barplot(data=skew_df, x=skew_df.index, y='Skewness')
plt.xticks(rotation=90)
plt.show();


# ### Findings
# 
# * The skewness values vary between fairly symmetrical (`Hillshade_3pm`) to highly skewed (`Vertical_Distance_To_Hydrlogy`). But in general I would say they are mostly moderately skewed.

# ## Categorical Data

# ### Wilderness Area

# In[ ]:


for c in wild_df.columns:
    print('{}: {}'.format(c, wild_df[c].value_counts()[1]))


# In[ ]:


tmpList = []
for c in wild_df.columns:
    tmpList += [str(c)] * wild_df[c].value_counts()[1]

se = pd.Series(tmpList)
df['Wilderness_Types'] = se.values


# In[ ]:


plt.figure(figsize=(16, 8))
sns.countplot(data=df, x='Wilderness_Types', hue='Cover_Type')
plt.show();


# ### Soil Type

# In[ ]:


for c in soil_df.columns:
    print('{}: {}'.format(c, soil_df[c].value_counts()[1]))

soil_df.sum().plot(kind='bar', figsize=(16, 8))
plt.title('Soil Type - Number of Observation')
plt.xlabel('Soil Type')
plt.ylabel('Number of Oberservation')
plt.xticks(rotation=90)
plt.show();
# In[ ]:


tmpList = []
for c in soil_df.columns:
    tmpList += [str(c)] * soil_df[c].value_counts()[1]

se = pd.Series(tmpList)
df['Soil_Types'] = se.values


# In[ ]:


plt.figure(figsize=(16, 8))
sns.countplot(data=df, x='Soil_Types', hue='Cover_Type')
plt.title('Number of Observation by Cover Type')
plt.xticks(rotation=90)
plt.show();


# In[ ]:


soil_df['Soil_Type29'].describe()


# In[ ]:


soil_df['Soil_Type29'].value_counts()


# In[ ]:


# sum Soil data values, and pass it as a series 
soil_sum = pd.Series(soil_df.sum())

# will sort values in descending order
soil_sum.sort_values(ascending = False, inplace = True)

# plot horizontal bar with given size using color defined
soil_sum.plot(kind='barh', figsize=(16, 12))

# horizontal bar flips columns in ascending order, this will filp it back in descending order
plt.gca().invert_yaxis()

plt.title('No. of observations of Soil Types')
plt.xlabel('No.of Observation')
plt.ylabel('Soil Types')

plt.xticks(rotation = 'horizontal')
plt.show();


# In[ ]:


soil_sum


# ### Findings
# 
# * The most positive observations are seen in `Soil_Type29` with 115,247 counts followed by `Soil_Type23`, `Soil_Type32` and `Soil_Type33`.
# * From a statistical standpoint, `Soil_Type29` is approximately `20%` represented in all of the data in regards to soil types.
# * The least found soil type in the data is `Soil_Type15`, `Soil_Type7` and `Soil_Type36`.
# * But based on the numbers from above we might consider dropping a handful soil types, eg. 7 and 8. We could do some fancy math but I will just pick the once which stand out to me.

# ## Feature Skewness (Categorical Data)

# In[ ]:


print(cat_df.skew())


# In[ ]:


skew_df = pd.DataFrame(cat_df.skew(), columns=['Skewness'])

sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(16, 8))
sns.barplot(data=skew_df, x=skew_df.index, y='Skewness')
plt.xticks(rotation=90)
plt.show();


# In[ ]:


# pick number of columns
ncol = 4
# make sure enough subplots
nrow = math.floor((len(cat_df.columns) + ncol - 1) / ncol)
# create the axes
height = 4 * nrow
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))

# go over a linear list of data
for i, col in enumerate(cat_df.columns):
    # compute an appropriate index (1d or 2d)
    ix = np.unravel_index(i, ax.shape) 

    sns.distplot(cat_df[col], ax=ax[ix])

plt.tight_layout()
plt.show();


# ### Findings
# 
# * `Soil_Type15` has the highest positive skewness value, meaning that the distribution is concentrated on the left side (`right skewed distribution`). 
# * But due to the nature of the data type all we can say is that some `Soil_Types` are less or more found during the sampling process. Either as `0` or `1`. 

# ## Class Distribution (Target)

# In[ ]:


covertype_df = pd.DataFrame(df.groupby('Cover_Type').size(), columns=['Size'])

plt.figure(figsize=(16, 8))
sns.barplot(data=covertype_df, x=covertype_df.index, y='Size')
plt.show();


# In[ ]:


# grouping by forest cover type and calculate the total occurance
df.groupby('Cover_Type').size()


# In[ ]:


t = df.groupby('Cover_Type').size()
print('Cover_Type 1 and 2 in percent: {:.2f}%'.format((t.values[0] + t.values[1]) / (df.shape[0] / 100)))


# ### Finding
# 
# * `Spruce` and `Lodgepole Pine` make up `85.22%` of the sampled trees.

# ## Continues data

# In[ ]:


# Box and whiskers plot
# Spread of numerical features

sns.set_style("whitegrid")
plt.subplots(figsize=(16, 12))

# Using seaborn to plot it horizontally
sns.boxplot(data=cont_df, orient='h', palette='pastel')

plt.title('Spread of data in Numerical Features')
plt.xlabel('Observation Distribution')
plt.ylabel('Features')
plt.show();


# In[ ]:


print('Elevation min/max: {} - {} meters'.format(df['Elevation'].min(), df['Elevation'].max()))


# ### Finding
# 
# * As seen in the graph above `Slope` for example is highly dense. But we need to take into account that we are compairing several diverent data here. It would seem that the `Slope` would have a very narrow spread, but thats not true.
# * Similarly, `Hillshade`'s features are dense in its range of values. 
# * In comparison `Horizontal_Distance_To_Roadways` and `Horizontal_Distance_To_Fire_Points` have huge range of values.
# 
# All observations displayed above represent their range of values based on the type of measurements, i.e. degree, meter etc. Which may or may not be somewhat misleading. We should correct that and plot the features base on their measurement types (meter, degree and index).

# ## Continues Data 

# **Feature Comparison**
# 
# Compare each feature in our data to our target variable, visualizing how much dense and distributed each target variable's class is compared to the feature. We will use Violin Plot to visualize this, a combination of Box Plot and Density Plot (Histogram).

# In[ ]:


# pick number of columns
ncol = 2
# make sure enough subplots
nrow = math.floor((len(cont_df.columns) + ncol - 1) / ncol)
# create the axes
height = 6 * nrow
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))

# go over a linear list of data
for i, col in enumerate(cont_df.columns):
    # compute an appropriate index (1d or 2d)
    ix = np.unravel_index(i, ax.shape) 

    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")

plt.tight_layout()
plt.show();


# ### Finding
# 
# * `Elevation` (meter) interestically differs very much by `Cover_Type`. For example, `Cover_tpye 7` is mainly seen in the higher elevations around 3400 meters.  Where as `Cover_Type 4`  can be found  between 2000 and about 2400 meters. 
# * `Elevation` seems the most important feature due it's variation acroos all `Cover_Type`'s.
# * `Aspect` (degrees azimuth[<sup>1</sup>](#fn1)) seems to have some concentration on either end of the spectrum. Most of them look like hourglasses.
# * `Slope` (degree) has no clear distinction between `Cover_Type`, some have a wider range than others. 
# * `Hillshade_9am` and `Hillshade_Noon` have a `negatively skewed distributions` with higher values between index 200-250 for most observations.
# * `Hillshade_3pm` has a `normal distribution`. for all classes.
# 

# ## Wilderness Areas

# In[ ]:


# pick number of columns
ncol = 2
# make sure enough subplots
nrow = math.floor((len(wild_df.columns) + ncol - 1) / ncol)
# create the axes
height = 6 * nrow
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))

# go over a linear list of data
for i, col in enumerate(wild_df.columns):
    # compute an appropriate index (1d or 2d)
    ix = np.unravel_index(i, ax.shape) 

    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")

plt.tight_layout()
plt.show();


# ### Finding
# 
# We are looking at values `0` and `1`. But we can observe that `Wilderness_Area1` and `Wilderness_Area3` are more dominant across more `Cover_Type`'s.

# ## Soil_Type

# In[ ]:


# pick number of columns
ncol = 2
# make sure enough subplots
nrow = math.floor((len(soil_df.columns) + ncol - 1) / ncol)
# create the axes
height = 6 * nrow
fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))

# go over a linear list of data
for i, col in enumerate(soil_df.columns):
    # compute an appropriate index (1d or 2d)
    ix = np.unravel_index(i, ax.shape) 

    sns.violinplot(data=df, x=df['Cover_Type'], y=col, ax=ax[ix], palette="coolwarm")

plt.tight_layout()
plt.show();


# ### Finding
# 
# * `Soil_Type4` is the only soil type that is present in all `Cover_Type`'s.
# * `Cover_Type 4` seems to have the least presence compared to all other soil types.

# ## Convariance matrix

# In[ ]:


# Set the style of the visualization
sns.set_style('white')

# Create a convariance matrix
corr = cont_df.corr()

# Generate a mask the size of our covariance matrix
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = None

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(16, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, sep=20, n=9, as_cmap=True)

# Draw the heatmapwith the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, mask=mask, square=True, annot=True)

plt.show();


# ### Finding 
# * Correlation findings in rough order:
#     * `Hillside_9am` - `Hillside_3pm`
#     * `Hillside_3pm` - `Aspect`
#     * `Hillside_9pm` - `Aspect`
#     * `Hillside_3pm` - `Hillshade_Noon`
#     * `Hillshade_Noon` - `Slope`
#     * `Horizontal_Distance_To_Hydrology` - `Vertical_Distance_To_Hydrology`

# ### Plot Correlation

# In[ ]:


corr_list = [['Hillshade_3pm', 'Aspect', ], 
             ['Hillshade_9am', 'Aspect', ], 
             ['Hillshade_3pm', 'Hillshade_9am'], 
             ['Hillshade_3pm', 'Hillshade_Noon'],
             ['Hillshade_Noon', 'Slope'],  
             ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']]


# In[ ]:


get_ipython().run_cell_magic('time', '', '# pick number of columns\nncol = 2\n# make sure enough subplots\nnrow = math.floor((len(corr_list) + ncol - 1) / ncol)\n# create the axes\nheight = 10 * nrow\nfig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=(20, height))\n\nk=0\nfor i, j in corr_list:\n    # compute an appropriate index (1d or 2d)\n    ix = np.unravel_index(k, ax.shape) \n    \n    sns.scatterplot(data=df, x = i, y = j, hue="Cover_Type", ax=ax[ix], \n                    legend = \'full\', palette=\'coolwarm\')\n\n    k += 1\n\nplt.tight_layout()\nplt.show();')


# ### Finding
# 
# * From the looks, it feels like that `Cover_Type 3` dominates most of the correlations.
# * `Hillshade_3pm` and `Hillshade_9am` look like sin curves 
# 

# ---

# # Confusion Metrix

# The confusion matrix represents the counts (or normalized counts) of our True Positives, False Positives, True Negatives and False Negatives. This can further be visualized when analyzing the effectiveness of our classification algorithm.
# 
# ![](confusion_matrix2.png)

# In[ ]:


#Load the data
cm_df = pd.read_csv('../input/covtype.csv')
# cm_df.drop('Id', axis=1, inplace=True)

#Define appropriate X and y
y = cm_df['Cover_Type']
X = cm_df.drop('Cover_Type', axis=1)

#Normalize the Data
for col in cm_df.columns:
    cm_df[col] = (cm_df[col]-min(cm_df[col]))/ (max(cm_df[col]) - min(cm_df[col]))

# Split the data into train and test sets.
X_cm_train, X_cm_test, y_cm_train, y_cm_test = train_test_split(X, y, random_state=0)

#Fit a model
classifier = KNeighborsClassifier(weights='distance', n_jobs=-1)
y_pred = classifier.fit(X_cm_train, y_cm_train).predict(X_cm_test)

#Create confusion matrix
cnf_matrix = confusion_matrix(y_pred, y_cm_test)


# In[ ]:


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_cm_test, y_pred)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure(figsize=(12, 12))
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure(figsize=(12, 12))
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show();

#----------------------------------------------
# Can be replaced by pandas_ml


# # Data Engineering

# * If not already done drop null values
# * If not already done delete duplicate entries (keep first)
# * Reduce features by keeping the most important once
# * Scale (normilize) data
# * Check if we have multiple entries across `Wilderness_Area` and `Soilt_Type` for each observation

# ## Checking Entries

# In[ ]:


wild_df.sum().sum() == df.shape[0]


# In[ ]:


soil_df.sum().sum() == df.shape[0]


# Both, `Wilderness_Area` and `Soil_Type` don't have multiple entries or any missing once. I believe.
# 
# To make sure we should iterate over the features using `iterrows`. But this will do. 
# 
# We already did check for null values and duplicate values.

# ## Feature Selection by Hand

# ### Dimentionality Reduction

# * From our data exploration we know that every feature has some sort of observations and therefore can't be deleted or should be at least.
# * Because we want to find which feature has an impact on the prediction, we let the mode do the work. But which one?
# * sklearn provides several classifer algorithims, like `Extra Tree`, `Random Forest`, `Gradient Boosting` and `AdaBoost` Classifiers which have an attribute called `feature_importance_`. This way we can see which feature is more important compare to others and by how much.[<sup>2</sup>](#fn2)
# * We will create a dataset with all ranked `feature_importances_` to pick the top 20.

# In[ ]:


# Drop added features first
df.drop(['Wilderness_Types', 'Soil_Types'], axis=1, inplace=True)


# In[ ]:


# Create a target (y) and feature (X) set
y = df['Cover_Type']
X = df.drop('Cover_Type', axis=1)


# In[ ]:


# Create an empty dataframe to hold our findings for feature_importances_
ranking_df = pd.DataFrame()


# ### Random Forest Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "RFC_model = RandomForestClassifier(random_state=0, n_jobs=-1)\nRFC_model.fit(X, y)\n\nimportances = RFC_model.feature_importances_\nindices = np.argsort(importances)[::-1]\n\n# Get feature name\nrfc_list = [X.columns[indices[f]] for f in range(X.shape[1])]\nranking_df['RFC'] = rfc_list\n\n# Get feature importance\nrfci_list = [importances[indices[f]] for f in range(X.shape[1])]\nranking_df['RFC importance'] = rfci_list")


# ### Ada Boost Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "ABC_model = AdaBoostClassifier(random_state=0)\nABC_model.fit(X, y)\n\nimportances = ABC_model.feature_importances_\nindices = np.argsort(importances)[::-1]\n\nabc_list = [X.columns[indices[f]] for f in range(X.shape[1])]\nranking_df['ABC'] = abc_list\n\nabci_list = [importances[indices[f]] for f in range(X.shape[1])]\nranking_df['ABC importance'] = abci_list")


# ### Gradient Boosting Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "GBC_model = GradientBoostingClassifier(random_state=0)\nGBC_model.fit(X, y)\n\nimportances = GBC_model.feature_importances_\nindices = np.argsort(importances)[::-1]\n\ngbc_list = [X.columns[indices[f]] for f in range(X.shape[1])]\nranking_df['GBC'] = gbc_list\n\ngbci_list = [importances[indices[f]] for f in range(X.shape[1])]\nranking_df['GBC importance'] = gbci_list")


# ### Extra Trees Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "ETC_model = ExtraTreesClassifier(random_state=0, n_jobs=-1)\nETC_model.fit(X, y)\n\nimportances = ETC_model.feature_importances_\nindices = np.argsort(importances)[::-1]\n\netc_list = [X.columns[indices[f]] for f in range(X.shape[1])]\nranking_df['ETC'] = etc_list\n\netci_list = [importances[indices[f]] for f in range(X.shape[1])]\nranking_df['ETC importance'] = etci_list")


# In[ ]:


ranking_df.head(25)


# ### Finding
# 
# Here we can see from each classifier the top 20 features.
# 
# * `Random Forest` and `Extra Tree` Classifier show the most similar results.
# * `Gradian Boosting` shows similar names just in a different order compared to `Random Forest` and `Extra Tree`
# * `AdaBoost` on the other hand shows an interesting and unique result. The top 8 feature alone are enough to make a good class prediction. Compare to all the other classifiers, here we have `Wilderness_Area4` on top, before `Elevation`.
# * `Elevation` dominates in all classifiers with a range of `18-65%`.
# * `Hillshade` features are seen in the top 20 in 3 out of 4 classifiers. `Random Forest` and `Extra Tree` Classifier show that `Hillshade` features having similar ranging.
# * `Horizontal_Distance_To_Hydrology` and `Vertical_Distance_To_Hydrology` are in all classifier top 10.
# * `Horizontal_Distance_To_Roadways` and `Horizontal_Distance_To_Fire_Points` are represented on the top in 3 out of 4 classifiers.
# * `Aspect` and `Slope` also show up in the top 20 across all classifiers, with the exception in `Gradian Boosting` `Slope` isn't in the top 20.
# * In regards to `Soil_Type` it is hard to find some commonality. Here are just a few:
#     * `Soil_Type2`
#     * `Soil_Type4`
#     * `Soil_Type10`
#     * `Soil_Type22`
#     * `Soil_Type23`
#     * `Soil_Type39`
#     
# 
# **Technical Note:**
# 
# We want to avoid `GradientBoostingClassifier` due to time it takes to run -> `10min 35s`

# Lets compare the top 25 features from `Random Forest` and `Extra Tree` Classifiers because they are the most similar classifiers.

# In[ ]:


ranking_df[['RFC','ETC']].head(25)


# From the listed features above we will select 20. This is more or less just based on intuition. But it seems appropriate to make a selection that rather than just go with one classifier, for exmaple `AdaBoost`.

# ### Feature Selection

# Here is a list of features I would choose.

# In[ ]:


sample_df = df[['Elevation', 
                'Aspect', 
                'Slope', 
                'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 
                'Horizontal_Distance_To_Roadways',
                'Horizontal_Distance_To_Fire_Points', 
                'Hillshade_9am', 
                'Hillshade_Noon',
                'Hillshade_3pm', 
                'Wilderness_Area1', 
                'Wilderness_Area3', 
                'Wilderness_Area4', 
                'Soil_Type2',
                'Soil_Type4', 
                'Soil_Type10', 
                'Soil_Type22', 
                'Soil_Type23', 
                'Soil_Type29',
                'Soil_Type39', 
                'Cover_Type']]


# ### Feature Scaling

# In[ ]:


y = df['Cover_Type']
X = sample_df.drop('Cover_Type', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ### Train-Test Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
print(X_train.shape, X_test.shape)


# ### K-Neighbors Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "clf = KNeighborsClassifier(weights='distance', n_jobs=-1)\nclf.fit(X_train, y_train)")


# ### Cross Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "accuracy = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs=-1)\nf1_score = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'f1_macro', n_jobs=-1)\n\nacc_mean = np.round(accuracy.mean() * 100, 2)\nf1_mean = np.round(f1_score.mean() * 100, 2)\n    \nprint('accuracy: {}%'.format(acc_mean))\nprint('f1_score: {}%'.format(f1_mean))")


# ### Finding
# 
# Let make a note of the accuracy and f1 score values and compare them with the workflow below. Remember, all we have done so far is trying to figure out "by hand" what features would be best to use for our prediction.
# 
# * `accuracy: 92.23%`
# * `f1_score: 86.74%`
# 
# For doing the ground work by hand so to speak nit bad. Actually pretty good. But it is a lot of work and can be very time consuming. 

# ---

# ## Feature Selection (the sklearn way)

# ### Feature Scaling

# > Feature scaling through standardization (or Z-score normalization) can be an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one.
# 
# 

# In[ ]:


y = df['Cover_Type']
X = df.drop('Cover_Type', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[ ]:


pd.DataFrame(data=X_scaled, columns=X.columns).head()


# ### Dimentionality Reduction with PCA
# 
# **Principal component analysis (PCA)**
# 
# > Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
# 
# Using Classifiers with its attribute `feature_importances_` is very time consuming and not particularly accurate at the end, due to the subjective matter how one chooses features. PCA does all the work for us and is much faster in doing it.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pca = PCA(n_components=20)\nX_pca = pca.fit_transform(X_scaled)')


# In[ ]:


pc = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
      'PC11', 'PC12', 'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 'PC18', 'PC19', 'PC20']

X_pca = pd.DataFrame(data=X_pca, columns=pc)
X_pca.head()


# **Note**: By using PCA we are loosing any meaningful feature names. All what we have done is tell PCA that we want to keep $n$ number of components (`n_components=20`).

# ### Train-Test Split

# We will go with the default test_size of 0.25 and set the random_state to 0 to be constant.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=0)

print(X_train.shape, X_test.shape)


# ### K-Neighbors Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "clf = KNeighborsClassifier(weights='distance', n_jobs=-1)\nclf.fit(X_train, y_train)")


# ### Cross Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "accuracy = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy', n_jobs=-1)\nf1_score = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'f1_macro', n_jobs=-1)")


# In[ ]:


acc_mean = np.round(accuracy.mean() * 100, 2)
f1_mean = np.round(f1_score.mean() * 100, 2)
    
print('accuracy: {}%'.format(acc_mean))
print('f1_score: {}%'.format(f1_mean))


# In[ ]:


predict = clf.predict(X_test)


# In[ ]:


# calculating accuracy
accuracy = accuracy_score(y_test, predict)

print('KNeighbors Classifier model')
print('Accuracy: {:.2f}%'.format(accuracy * 100))


# In[ ]:


knn_classification_report = classification_report(y_test, predict)
print(knn_classification_report)


# ### Finding
# 
# Now lets remember the values we got when we did all the work to find the right features by hand:
# 
# * `accuracy: 92.23%`
# * `f1_score: 86.74%`
# 
# Using PCA to reduce dimentionality instead gives use the following values:
# 
# * `accuracy: 90.72%`
# * `f1_score: 84.44%`
# 
# Now, is that a big enough difference to justify all the labour (time mainly) to get such a small gain of ~2%? Properbly not.

# ---

# # Model Evaluation

# We will use the train_test_split data from previous steps (6.3.). And we shall use the follow classifiers from `sklearn.ensemble`:
# 
# ```
# 1. Random Forest Classifier (RFC)
# 2. Stochastic Gradient Descent Classifier (SGDC)
# 3. Extraa Trees Classifier (ETC)
# ```

# ## Random Forest Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', 'rf_clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)\nrf_clf.fit(X_train, y_train)')


# ### Cross-Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "accuracy = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)\nf1_score = cross_val_score(rf_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)")


# In[ ]:


print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))
print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))


# ## Extra Tree Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', 'tree_clf = ExtraTreesClassifier(n_estimators=50, random_state=0, n_jobs=-1)\ntree_clf.fit(X_train, y_train)')


# ### Cross Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "accuracy = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)\nf1_score = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)")


# In[ ]:


print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))
print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))


# ## Gradient Boosting Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', 'grad_clf = GradientBoostingClassifier(n_estimators=50, random_state=0)\ngrad_clf.fit(X_train, y_train)')


# ### Cross Validation

# In[ ]:


get_ipython().run_cell_magic('time', '', "accuracy = cross_val_score(grad_clf, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)\nf1_score = cross_val_score(grad_clf, X_train, y_train, cv=10, scoring='f1_macro', n_jobs=-1)")


# In[ ]:


print('accuracy: {:.2f}%'.format(accuracy.mean() * 100))
print('f1_score: {:.2f}%'.format(f1_score.mean() * 100))


# ## Finding

# * So `GradientBoostingClassifier` has the longest run time of all classifiers, building the model took `4min` and the cross validation, og my god, took (Wall time) `35min 49s`. Not only it took the longest, but the accuracy and f1 score weren't that great, with `74.94%` and `60.19%` respectively.
# * Again, the `CPU time` is the first observation which stands out. Mainly the time spend on `cross_val_score`. 
# * `KNeighborsClassifier` takes mere `40s` where as in contrast `RandomForestClassifier` takes `13min` to calculate (Wall time).
# * The overall best result comes from `ExtraTreesClassifier`, the build of the model and the cross validation took `3min 24s`. With an accuracy of `94.21%` and a f1 score of `89.91%`.
# * There is one question though I would like to have anwsered, and that is 
# > Shall we use `cross_val_score` or `cross_validate`, or better when to use one over the other?
# 
# 

# # Testing Model

# We will test our model using the `Random Forest Classifier` model.

# In[ ]:


from sklearn.metrics import precision_recall_fscore_support


# In[ ]:


est = [10, 25, 50, 100, 150, 200, 250]

for e in est:
    clf = RandomForestClassifier(n_estimators=e, random_state=0, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    
    predict = clf.predict(X_test)
    
    print('n_estimators={}'.format(e))
    
    accuracy = accuracy_score(y_test, predict)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    p, r, f, s = precision_recall_fscore_support(y_test, predict, average='weighted')
    print('fscore: {:.2f}%'.format(f*100))
    
    print()


# In[ ]:


clf = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
clf = clf.fit(X_train, y_train)


# In[ ]:


# predicting unseen data
predict = clf.predict(X_test)


# In[ ]:


# calculating accuracy
accuracy = accuracy_score(y_test, predict)

print('Random Forest Classifier model')
print('Accuracy: {:.2f}%'.format(accuracy * 100))


# In[ ]:


print(classification_report(y_test, predict))


# ## Finding
# 
# Iterating over a list of estimator numbers [10, 25, 50, 100, 150, 200, 250] to find the optimal value for the `RandomForestClassifier`. Currently the default value is `10` and will change to `100` in the upcoming version 0.22.
# 
# n_estimator | accuracy | f_score
# ------------|----------|--------
# 10          | 92.70%   | 92.65%
# 25          | 93.57%   | 93.54%
# 50          | 93.86%   | 93.82%
# 100         | 93.97%   | 93.94%
# 150         | 94.02%   | 93.99%
# 200         | 94.03%   | 94.00%
# 250         | 94.06%   | 94.02%
# 
# I saddled on `n_estimators = 50` because it seems a good compromise between `10` and `100`. Higher number give us a better, a more accurate performance. On the other hand a higher has the drawback of increased processing power and can take a lot of time.
# 
# 

# # Pipeline using seperate Train-Test datasets

# ## Read Data

# In[ ]:


df = pd.read_csv('../input/covtype.csv')
df.head()


# In[ ]:


train_df, test_df = train_test_split(df, test_size=0.1)
print(train_df.shape, test_df.shape)


# ## Grid Search using Pipeline

# In[ ]:


y = train_df['Cover_Type']
X = train_df.drop('Cover_Type', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Create the pipeline
pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA(iterated_power=7)),
                 ('clf', ExtraTreesClassifier(random_state=0, n_jobs=-1))])

param_range = [1, 2, 3, 4, 5]

# Create the grid parameter
grid_params = [{'pca__n_components': [10, 15, 20, 25, 30],
                'clf__criterion': ['gini', 'entropy'],
                'clf__min_samples_leaf': param_range,
                'clf__max_depth': param_range,
                'clf__min_samples_split': param_range[1:]}]

# Create the grid, with "pipe" as the estimator
gridsearch = GridSearchCV(estimator=pipe,
                          param_grid=grid_params,
                          scoring='accuracy',
                          cv=3)

# Fit using grid search
gridsearch.fit(X_train, y_train)


# **The following result is from the sandbox notebook**
GridSearchCV(cv=3, error_score='raise-deprecating',
       estimator=Pipeline(memory=None,
     steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), ('pca', PCA(copy=True, iterated_power=7, n_components=None, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
           max_dep...mators='warn', n_jobs=-1,
           oob_score=False, random_state=0, verbose=0, warm_start=False))]),
       fit_params=None, iid='warn', n_jobs=None,
       param_grid=[{'pca__n_components': [10, 15, 20, 25, 30], 'clf__criterion': ['gini', 'entropy'], 'clf__min_samples_leaf': [1, 2, 3, 4, 5], 'clf__max_depth': [1, 2, 3, 4, 5], 'clf__min_samples_split': [2, 3, 4, 5]}],
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='accuracy', verbose=0)
# In[ ]:


# Best accuracy
print('Best accuracy: %.3f' % gridsearch.best_score_)

# Best params
print('\nBest params:\n', gridsearch.best_params_)


# **The following result is from the sandbox notebook**
Best accuracy: 0.601

Best params:
 {'clf__criterion': 'entropy', 'clf__max_depth': 5, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 4, 'pca__n_components': 25}
# ## Pipeline using GridSearchCV Results

# In[ ]:


pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA(n_components=25)),
                 ('tree', ExtraTreesClassifier(criterion='entropy',
                                               max_depth=5, 
                                               min_samples_leaf=1, 
                                               min_samples_split=4,
                                               random_state=42, 
                                               n_jobs=-1))])


# In[ ]:


y = test_df['Cover_Type']
X = test_df.drop('Cover_Type', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


score = pipe.score(X_test, y_test)
print('Test Accuarcy: {:.2f}%'.format(score * 100))


# In[ ]:


cross_val_score(pipe, X_train, y_train)


# In[ ]:


y_pred = pipe.predict(X_test)
print("Testing Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


# ### Finding

# Using GridSearchCV didn't help the result. But that is most likely due to inexperience. 

# ## Pipeline

# In[ ]:


pipe = Pipeline([('scl', StandardScaler()),
                 ('pca', PCA(n_components=20)),
                 ('tree', ExtraTreesClassifier(random_state=42, n_jobs=-1))])


# In[ ]:


pipe.fit(X_train, y_train)


# In[ ]:


score = pipe.score(X_test, y_test)
print('Test Accuarcy: {:.2f}%'.format(score * 100))


# In[ ]:


cross_val_score(pipe, X_train, y_train)


# In[ ]:


y_pred = pipe.predict(X_test)
print("Testing Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))


# # Conclusion

# With all the work done now, can we we actally predict which `Cover Type` we will see, given for example `Elevation`, `Slope`, `Hillshade`, `Soil_Type` and a few more? 
# 
# **Yes**! With the given data collected by the researchers we can predict which type of tree could we find in a given area.
# 
# **Technical Note:**
# 
# `Extra Tree Classifier` is the best choice to make predictions for the `Forest Cover Type Dataset`. Two points stand out here:
# 
# * calculation time
# * accuracy and f1_score
# 

# # Footnotes

# <span id="fn1"> (1)  
# > An azimuth is an angular measurement in a spherical coordinate system. The vector from an observer to a point of interest is projected perpendicularly onto a reference plane; the angle between the projected vector and a reference vector on the reference plane is called the azimuth.
# 
# ![](Azimuth-Altitude_schematic.svg)

# <span id="fn2"> (2) From [stackoverflow](https://stackoverflow.com/a/20187191) </span>
# > `ExtraTreeClassifier` is an extremely randomized version of `DecisionTreeClassifier` meant to be used internally as part of the `ExtraTreesClassifier` ensemble.
# >
# > Averaging ensembles such as a `RandomForestClassifier` and `ExtraTreesClassifier` are meant to tackle the variance problems (lack of robustness with respect to small changes in the training set) of individual `DecisionTreeClassifier` instances.
# >
# > If your main goal is maximizing prediction accuracy you should almost always use an ensemble of decision trees such as `ExtraTreesClassifier` (or alternatively a boosting ensemble) instead of training individual decision trees.

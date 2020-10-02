#!/usr/bin/env python
# coding: utf-8

# # Import the Libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from collections import Counter

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib 
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from keras import Sequential
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from keras.utils import to_categorical #Used for multi-class setting
from keras.utils import np_utils
from keras.optimizers import SGD


# # Load the Data

# In[ ]:


train_data = pd.read_csv("../input/airplane-accidents-severity-dataset/train.csv")


# In[ ]:


train_data.shape


# In[ ]:


train_data.head()


# In[ ]:


#These are our label columns.
train_data['Severity'].unique()


# In[ ]:


train_data.columns


# <h>Description of columns</h>
# <ol>
#     <li>Severity : 	a description (4 level factor) on the severity of the crash [Target]</li>
#     <li>Safety_Score : a measure of how safe the plane was deemed to be</li>
#     <li>Days_Since_Inspection : how long the plane went without inspection before the incident</li>
#     <li>Total_Safety_Complaints : number of complaints from mechanics prior to the accident</li>
#     <li>Control_Metric : an estimation of how much control the pilot had during the incident given the factors at play</li>
#     <li>Turbulence_In_gforces : the recorded/estimated turbulence experienced during the accident</li>
#     <li>Cabin_Temperature : the last recorded temperature before the incident, measured in degrees fahrenheit</li>
#     <li>Accident_Type_Code : the type of accident (factor, not numeric)</li>
#     <li>Max_Elevation : Description not provided</li>
#     <li>Violations : Number of violations that the aircraft received during inspections</li>
#     <li>Adverse_Weather_Metric : Description not provided</li>
#     <li>Accident_ID : unique id assigned to each row</li>
# </ol>
# 

# In[ ]:


train_data.head()


# In[ ]:


train_data_for_pair_plot = train_data[["Severity", "Safety_Score", "Days_Since_Inspection",                                       "Total_Safety_Complaints", "Control_Metric", "Turbulence_In_gforces",                                      "Cabin_Temperature"]]
sns.set_style("whitegrid")
sns.pairplot(train_data_for_pair_plot, hue="Severity", height=3, diag_kind = "kde")
plt.show()


# <b> Here the problem with pairplot is that it is creating scatter plot for 2 continuous features hence it is impossible to understand much from it</b>

# # Custom Functions

# In[ ]:


def draw_boxplot(feature_under_observation, dataset):
    """Remember that to visuaslize median,percentile,IQR(Inter Quartile Range) box-plots are best."""
    ax = sns.boxplot(x="Severity", y=feature_under_observation, data=dataset)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.show()
    
    
def draw_violinplot(feature_under_observation, dataset):
    """Remember that to visuaslize median,percentile,IQR(Inter Quartile Range) box-plots are best."""
    ax = sns.violinplot(x="Severity", y=feature_under_observation, data=dataset)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.show()

    
def normalize(dataframe):
    #print("Here")
    test = dataframe.copy()
    for col in test.columns:
        if(col != "Accident_ID" and col !="Severity"):
            max_val = max(dataframe[col])
            min_val = min(dataframe[col])
            test[col] = (dataframe[col] - min_val) / (max_val-min_val)
    return test


# # EDA

# ## Distribute different class data for EDA 

# In[ ]:


train_data["Severity"].unique()


# In[ ]:


severity_1 = train_data.loc[train_data["Severity"] == "Minor_Damage_And_Injuries"]

severity_2 = train_data.loc[train_data["Severity"] == "Significant_Damage_And_Fatalities"]

severity_3 = train_data.loc[train_data["Severity"] == "Significant_Damage_And_Serious_Injuries"]

severity_4 = train_data.loc[train_data["Severity"] == "Highly_Fatal_And_Damaging"]


# ## 'Severity' Column

# In[ ]:


#Severity is our target label.
train_data['Severity'].value_counts().values


# In[ ]:


total = len(train_data)*1
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=train_data['Severity'].value_counts().index,            y=train_data['Severity'].value_counts().values)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
for p in ax.patches:
    """ax.patches gives a list of rectangle object. Each element represent a rectangle in above histogram.
    Example: Rectangle(xy=(-0.4, 0), width=0.8, height=1541, angle=0)"""
    ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))
plt.title("Number of favourite dishes")
plt.xlabel("Count")
plt.ylabel("Name")


# <h>We can see that dataset is not skewed.<br>
# Since class distribution is not imbalanced.</h>

# ## 'Safety_Score' Column

# In[ ]:


#A measure of how safe the plane was deemed to be
safety_score_data = train_data["Safety_Score"]


# In[ ]:


safety_score_data.head()


# In[ ]:


safety_score_data.describe()


# In[ ]:


sns.distplot(safety_score_data)


# In[ ]:


sns.boxplot(x='Safety_Score', data=train_data)


# <b>It shows outliers. Maybe with safety score > 83 </b> 

# In[ ]:


up = train_data["Safety_Score"].mean()+3*train_data["Safety_Score"].std()
low = train_data["Safety_Score"].mean()-3*train_data["Safety_Score"].std()
print(len(train_data[(train_data["Safety_Score"]>up)]))
print(len(train_data[(train_data["Safety_Score"]<low)]))


# In[ ]:


draw_boxplot("Safety_Score", train_data)


# <b>More then 25% of highly fatal accident have a safety score in range 20-30. One interersting thing to notice is the outliers on the basis of severity</b>

# In[ ]:


draw_violinplot("Safety_Score", train_data)


# <b>The median of Highly_Fatal severity has lowest Safety_Score</b>

# ## 'Days_Since_Inspection' Column

# In[ ]:


#how long the plane went without inspection before the incident
days_since_inspection_data = train_data['Days_Since_Inspection']


# In[ ]:


days_since_inspection_data.head()


# In[ ]:


days_since_inspection_data.describe()


# In[ ]:


sns.distplot(days_since_inspection_data)


# In[ ]:


sns.boxplot(x='Days_Since_Inspection', data=train_data)


# In[ ]:


up = train_data["Days_Since_Inspection"].mean()+3*train_data["Days_Since_Inspection"].std()
low = train_data["Days_Since_Inspection"].mean()-3*train_data["Days_Since_Inspection"].std()
print(len(train_data[(train_data["Days_Since_Inspection"]>up)]))
print(len(train_data[(train_data["Days_Since_Inspection"]<low)]))


# In[ ]:


draw_boxplot("Days_Since_Inspection", train_data)


# In[ ]:


draw_violinplot("Days_Since_Inspection", train_data)


# ## 'Total_Safety_Complaints' Column

# In[ ]:


#number of complaints from mechanics prior to the accident
total_safety_complaints = train_data['Total_Safety_Complaints']


# In[ ]:


total_safety_complaints.head()


# In[ ]:


total_safety_complaints.describe()


# In[ ]:


sns.distplot(total_safety_complaints)


# In[ ]:


sns.boxplot(x="Total_Safety_Complaints", data=train_data)


# In[ ]:


up = train_data["Total_Safety_Complaints"].mean()+3*train_data["Total_Safety_Complaints"].std()
low = train_data["Total_Safety_Complaints"].mean()-3*train_data["Total_Safety_Complaints"].std()
print(len(train_data[(train_data["Total_Safety_Complaints"]>up)]))
print(len(train_data[(train_data["Total_Safety_Complaints"]<low)]))


# In[ ]:


draw_boxplot("Total_Safety_Complaints", train_data)


# In[ ]:


draw_violinplot("Total_Safety_Complaints", train_data)


# ## 'Control_Metric' Column

# In[ ]:


# an estimation of how much control the pilot had during the incident given the factors at play
control_metric_data = train_data['Control_Metric']


# In[ ]:


control_metric_data.head()


# In[ ]:


control_metric_data.describe()


# In[ ]:


sns.distplot(control_metric_data)


# In[ ]:


sns.boxplot(x="Control_Metric", data=train_data)


# In[ ]:


up = train_data["Control_Metric"].mean()+3*train_data["Control_Metric"].std()
low = train_data["Control_Metric"].mean()-3*train_data["Control_Metric"].std()
print(len(train_data[(train_data["Control_Metric"]>up)]))
print(len(train_data[(train_data["Control_Metric"]<low)]))


# In[ ]:


draw_boxplot("Control_Metric", train_data)


# In[ ]:


draw_violinplot("Control_Metric", train_data)


# ## 'Turbulence_In_gforces' Column

# In[ ]:


# the recorded/estimated turbulence experienced during the accident
turbulence_in_gforces_data =train_data['Turbulence_In_gforces']


# In[ ]:


turbulence_in_gforces_data.head()


# In[ ]:


turbulence_in_gforces_data.describe()


# In[ ]:


sns.distplot(turbulence_in_gforces_data)


# In[ ]:


sns.boxplot(x="Turbulence_In_gforces", data=train_data)


# In[ ]:


up = train_data["Turbulence_In_gforces"].mean()+3*train_data["Turbulence_In_gforces"].std()
low = train_data["Turbulence_In_gforces"].mean()-3*train_data["Turbulence_In_gforces"].std()
print(len(train_data[(train_data["Turbulence_In_gforces"]>up)]))
print(len(train_data[(train_data["Turbulence_In_gforces"]<low)]))


# In[ ]:


draw_boxplot("Turbulence_In_gforces", train_data)


# In[ ]:


draw_violinplot("Turbulence_In_gforces", train_data)


# ## 'Cabin_Temperature' Column

# In[ ]:


#the last recorded temperature before the incident, measured in degrees fahrenheit
cabin_temperature_data = train_data['Cabin_Temperature']


# In[ ]:


cabin_temperature_data.head()


# In[ ]:


cabin_temperature_data.describe()


# In[ ]:


sns.distplot(cabin_temperature_data)


# In[ ]:


sns.boxplot(x="Cabin_Temperature", data=train_data)


# In[ ]:


up = train_data["Cabin_Temperature"].mean()+3*train_data["Cabin_Temperature"].std()
low = train_data["Cabin_Temperature"].mean()-3*train_data["Cabin_Temperature"].std()
print(len(train_data[(train_data["Cabin_Temperature"]>up)]))
print(len(train_data[(train_data["Cabin_Temperature"]<low)]))


# In[ ]:


draw_boxplot("Cabin_Temperature", train_data)


# In[ ]:


draw_violinplot("Cabin_Temperature", train_data)


# ## 'Accident_Type_Code' Column

# In[ ]:


#the type of accident (factor, not numeric)
accident_type_code_data = train_data['Accident_Type_Code']


# In[ ]:


accident_type_code_data.head()


# In[ ]:


accident_type_code_data.value_counts().apply(lambda x: x/sum(accident_type_code_data.value_counts()) * 100)


# In[ ]:


sns.countplot(x="Accident_Type_Code", data=train_data)


# ## 'Max_Elevation' Column

# In[ ]:


max_elevation_data = train_data['Max_Elevation']


# In[ ]:


max_elevation_data.head()


# In[ ]:


max_elevation_data.describe()


# In[ ]:


sns.distplot(max_elevation_data)


# In[ ]:


sns.boxplot(x="Max_Elevation", data=train_data)


# In[ ]:


up = train_data["Max_Elevation"].mean()+3*train_data["Max_Elevation"].std()
low = train_data["Max_Elevation"].mean()-3*train_data["Max_Elevation"].std()
print(len(train_data[(train_data["Max_Elevation"]>up)]))
print(len(train_data[(train_data["Max_Elevation"]<low)]))


# In[ ]:


draw_boxplot("Max_Elevation", train_data)


# In[ ]:


draw_violinplot("Max_Elevation", train_data)


# ## 'Violations' Column

# In[ ]:


violations_data = train_data['Violations']


# In[ ]:


violations_data.unique()


# In[ ]:


violations_data.value_counts().apply(lambda x: x/sum(violations_data.value_counts()) * 100)


# In[ ]:


violations_data.head()


# In[ ]:


violations_data.describe()


# In[ ]:


sns.countplot(x="Violations", data=train_data)


# ## 'Adverse_Weather_Metric' Columns

# In[ ]:


adverse_weather_metric_data = train_data['Adverse_Weather_Metric']


# In[ ]:


adverse_weather_metric_data.head()


# In[ ]:


adverse_weather_metric_data.describe()


# In[ ]:


sns.distplot(adverse_weather_metric_data)


# In[ ]:


sns.boxplot(x="Adverse_Weather_Metric", data=train_data)


# In[ ]:


up = train_data["Adverse_Weather_Metric"].mean()+3*train_data["Adverse_Weather_Metric"].std()
low = train_data["Adverse_Weather_Metric"].mean()-3*train_data["Adverse_Weather_Metric"].std()
print(len(train_data[(train_data["Adverse_Weather_Metric"]>up)]))
print(len(train_data[(train_data["Adverse_Weather_Metric"]<low)]))


# In[ ]:


draw_boxplot("Adverse_Weather_Metric", train_data)


# In[ ]:


severity_1.head()


# In[ ]:


draw_violinplot("Adverse_Weather_Metric", train_data)


# ## 'Accident_ID' Column

# In[ ]:


accident_id_data = train_data['Accident_ID']


# In[ ]:


accident_id_data.head()


# In[ ]:


accident_id_data.duplicated().value_counts()
#Every ID is unique


# In[ ]:


sns.distplot(accident_id_data)


# ## Multivariate Analysis

# ### Quantitative vs Quantitaive

# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap="RdYlGn")


# In[ ]:


sns.pairplot(train_data.drop(['Accident_Type_Code', 'Violations'], axis=1), hue='Severity')


# In[ ]:


#Earlier you have created box-plot for every severity class for each variable


# ### Quantitative vs Categorical

# <b>When we want to analyze a quantitative variable in two categorical dimensions at once, there is a suitable function for this in the seaborn library called catplot() <b>

# In[ ]:


sns.catplot(x='Severity', y='Adverse_Weather_Metric', col='Violations',
               data=train_data, kind="box");


# ### Categorical vs Categorical

# In[ ]:


sns.countplot(x="Accident_Type_Code", hue="Severity", data=train_data)


# In[ ]:


sns.countplot(x="Violations", hue="Severity", data=train_data)


# #### T-SNE

# In[ ]:


normalized_data = normalize(train_data)


# In[ ]:


normalized_data


# In[ ]:


model = TSNE(perplexity=50)
train_tsne_data = model.fit_transform(normalized_data.                                      drop(["Accident_ID","Severity"], axis=1))


# In[ ]:


data_y = train_data["Severity"]


# In[ ]:


#Byte tsne data will return 2-dimesnions.
x_ax = train_tsne_data[:,0]
y_ax = train_tsne_data[:,1]


# In[ ]:


#Plot on the basis of severity
plt.figure(figsize=(16,10))
sns.scatterplot(x=x_ax, y=y_ax, hue=data_y, palette=sns.color_palette("hls", 4), legend="full")
plt.show()


# In[ ]:


#Plot on the basis of Accident_Type_Code
plt.figure(figsize=(16,10))
sns.scatterplot(x=x_ax, y=y_ax, hue=train_data['Accident_Type_Code'].map({1: 'orange', 
                                                                2: 'blue',
                                                                3: 'red',
                                                                4: 'yellow',
                                                                5: 'green',
                                                                6: 'black',
                                                                7: 'purple'}), legend="full")
plt.show()


# In[ ]:


#Plot on the basis of Violations
plt.figure(figsize=(16,10))
sns.scatterplot(x=x_ax, y=y_ax, hue=train_data['Violations'].map({0: 'orange', 
                                                                1: 'blue',
                                                                2: 'red',
                                                                3: 'yellow',
                                                                4: 'green',
                                                                5: 'black'}), legend="full")
plt.show()


# In[ ]:


train_data[['Accident_Type_Code', 'Violations','Severity']].groupby('Severity').mean().plot()


# # New Features

# In[ ]:


#Here we will create new columns for Accident_Type_Code and Violation
Accident_Type_Code_Dummies = pd.get_dummies(train_data["Accident_Type_Code"], prefix="Accident_Type_Code")
train_data = train_data.join(Accident_Type_Code_Dummies)
Violations_Dummies = pd.get_dummies(train_data["Violations"], prefix="Violations")
train_data = train_data.join(Violations_Dummies)
new_train_data = train_data.drop(["Accident_Type_Code"], axis=1)
new_train_data = new_train_data.drop(["Violations"], axis=1)


# In[ ]:


new_train_data.shape


# In[ ]:


#Now we have to see EDA of these new features


# In[ ]:


new_train_data['Violations_0'].value_counts()


# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True,cmap="RdYlGn")


# In[ ]:


corr_matrix.index


# In[ ]:


corr_matrix["Severity"].sort_values(ascending=False)


# Nothing is too strongly correlated with our target variable.

# In[ ]:


for col in train_data.columns:
    if(col!='Severity'):
        sns.jointplot(x=train_data[col], y=train_data["Severity"])
        plt.show()


# <ol>
# <li>Total_Safety_Complaints</li>
# <li>Control_Metric</li>
# <li>Cabin_Temperature</li>
# <li>Adverse_Weather_metric</li>
# </ol>
# They are showing few outliers

# In[ ]:


for col in train_data.columns:
    if(col!='Severity'):
        sns.distplot(train_data[col])
        plt.show()


# <ol>
# <li>Total_Safety_Complaints</li>
# <li>Adverse_Weather_metric</li>
# </ol>
# They are showing right skewness

# In[ ]:


print(train_data['Safety_Score'].skew())
print(train_data['Days_Since_Inspection'].skew())
print(train_data['Total_Safety_Complaints'].skew())
print(train_data['Control_Metric'].skew())
print(train_data['Turbulence_In_gforces'].skew())
print(train_data['Cabin_Temperature'].skew())
print(train_data['Max_Elevation'].skew())
print(train_data['Adverse_Weather_Metric'].skew())


# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


corr_matrix["Severity"].sort_values(ascending=False)


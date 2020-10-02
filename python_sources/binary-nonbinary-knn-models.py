#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries for data analysis, visualization, and ML model building
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Exploratory Data Analysis

# In[ ]:


train = pd.read_csv('/kaggle/input/learn-together/train.csv')
test = pd.read_csv('/kaggle/input/learn-together/test.csv')
sample_submission = pd.read_csv('/kaggle/input/learn-together/sample_submission.csv')


# In[ ]:


#List of all non-binary columns
non_binary_cols = ['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']


# In[ ]:


#The 'aspect' variable reflects the azimuth measurement in degrees (0-360).
#Since a reading of 180 degrees indicates that the sun is directly overhead,
#readings of 200 and 160 degrees would be equivalent. 
#Therefore, I updated the column for this variable to situate all readings between 0 and 180.
train['Aspect']=np.absolute(180-train['Aspect'])
test['Aspect']=np.absolute(180-test['Aspect'])


# In[ ]:


def create_corr_viz(x):
    #Create correlation table
    correlation_table = x.corr().round(3)
    #Diagonally slice table to remove duplicates using np.tril method
    data_viz_cols = correlation_table.columns
    data_viz_index = correlation_table.index
    correlation_table = pd.DataFrame(np.tril(correlation_table),columns=data_viz_cols,index=data_viz_index).replace(0,np.nan).replace(1,np.nan)
    #Adjust plot size
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.heatmap(correlation_table, annot=True,cmap='coolwarm')
    plt.show()
create_corr_viz(train[non_binary_cols])


# The three non-binary variables with the biggest *positive* correlation coefficients are as follows:
# 
# -----------
# | Feature 1 | Feature 2 | Correlation |
# |---|---|---|
# |Vertical Distance to Hydrology | Horizontal Distance to Hydrology|0.65|
# |Hillshade Noon|Hillshade 3pm |0.61|
# |Horizontal Distance to Roadways | Elevation |0.58|
# 
# ----------
# 
# The three non-binary variables with the biggest *negative* correlation coefficients are as follows:
# 
# -----------
# | Feature 1 | Feature 2 | Correlation |
# |---|---|---|
# |Hillshade 9am| Hillshade 3pm|-0.78|
# |Hillshade Noon|Slope|-0.61|
# |Hillshade Noon|Aspect|-0.58|
# 
# ----------
# Observations Regarding Feature Correlations:
# 
# * No features are strongly correlated with the target variable, Cover Type. Horizontal Distance to Roadways is negatively-correlated with the target with a weak -0.11 coefficient. There is a good chance non of these features will be useful in building the model.
# * One would expect to see a positive correlation between Hillshade Noon and Slope as well as a negative  correlation between Hillshade Noon and Aspect, as Slope and Aspect would be the primary if not exclusive variables calculating the degree of hillshade when the Sun is directly overhead. 
# * The strong correlation between the Vertical and Horizontal Distances to Hydrology seems to suggest that they are measuring proximity to the *same* body of water and can be combined using the Pythagorean theorem to calculate a direct distance to the water body. 
# * Horizontal Distance to Roadways and Elevation is an intuitive correlation, as is the negative correlation between Hillshade 9am and Hillshade 3pm. 

# In[ ]:


#Create lists representing feature and target variables
features = ['Elevation', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
target = ['Cover_Type']

#Create Function to Perform the Following Two Tasks:
#1.Scale training set values to values between 0 and 1
#2.Split train model into train, test split for supervised learning
def scale_n_split(df,features,target):
    scale = MinMaxScaler((0,1))
    scaled_train = scale.fit_transform(df[features])
    scaled_train_df = pd.DataFrame(scaled_train,columns=features)
    scaled_train_df[target] = train[target]
    supervised_train,supervised_test = train_test_split(scaled_train_df,random_state=42)
    return supervised_train,supervised_test

supervised_train,supervised_test = scale_n_split(train,features,target)


# In[ ]:


#Train KNN Classifier Model inclusive of all non-binary variables

def train_knn(train_df,test_df,features,target):
    model = KNeighborsClassifier(n_neighbors=3,algorithm='auto')
    model.fit(train_df[features],train_df[target])
    predictions = model.predict(test_df[features])
    score = accuracy_score(predictions,test_df[target])
    score.round(2)
    return score,predictions,model

score,predictions,model = train_knn(supervised_train,supervised_test,features,target)
cross_val_scores = cross_val_score(model,supervised_train[features],supervised_train[target],cv=5)
cross_val_scores
#pd.Series(predictions).value_counts(dropna=False,ascending=False)


# In[ ]:


test['Cover_Type'] = model.predict(test[features])
test['Cover_Type'].value_counts(dropna=False)
#test[['Id','Cover_Type']].to_csv('knn_init_submission.csv',index=False)


# In[ ]:


binary_cols = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',]

binary_train,binary_test = scale_n_split(train,binary_cols,target)
binary_score,binary_predictions,binary_model = train_knn(binary_train,binary_test,binary_cols,target)
cross_val_scores = cross_val_score(binary_model,binary_train[binary_cols],binary_train[target],cv=5)
cross_val_scores


# In[ ]:


#test['Cover_Type'] = binary_model.predict(test[binary_cols])


# In[ ]:


#submission_binary_cols = test[['Id','Cover_Type']]
#submission_binary_cols.to_csv('submission_2.csv',index=False)


# In[ ]:


train.head()


# In[ ]:


target_corrs = round(abs(train.corr()[target]),2)
significant_cols = [index for index,val in target_corrs['Cover_Type'].iteritems() if val>0.1]


# In[ ]:


soil_type_cols = [x for x in significant_cols if 'Soil' in x ]
all_soil_type_cols = [x for x in train.columns if 'Soil' in x ]


# In[ ]:


soil_frequencies = pd.Series(train[soil_type_cols].sum()/train.shape[0])
soil_frequencies.sort_values(ascending=False)


# In[ ]:


sum(soil_frequencies)


# In[ ]:


target_corrs.sort_values(by='Cover_Type',ascending=False)


# In[ ]:


train[all_soil_type_cols].where(train[all_soil_type_cols].sum(axis=1)>1).any()


# In[ ]:


new_model = KNeighborsClassifier(n_neighbors=3)
scale = MinMaxScaler((0,1))
scaled_train = scale.fit_transform(train[features])
scaled_train_df = pd.DataFrame(scaled_train,columns=features)
scaled_train_df[target] = train[target]
new_model.fit(scaled_train_df[features],scaled_train_df[target])
predictions = model.predict(test[features])


# In[ ]:


test['Cover_Type'] = predictions


# In[ ]:


submission_retry_knn =  test[['Id','Cover_Type']]
submission_retry_knn.to_csv('submission_5.csv',index=False)


# In[ ]:





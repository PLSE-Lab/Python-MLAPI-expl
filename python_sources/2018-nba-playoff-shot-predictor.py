#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# Hi Guys,
# 
# This notebook is a simple machine learning model to predict the outcome of a shot in the playoffs based on a number of factors in the dataset (Player, Location, Shot Type, etc.). The approach is fairly simple, but it shows the power of machine learning and a random forest classifier to predict very dynamic events. Any feedback or comment is appreciate it! 

# # **Ingest**

# In[ ]:


#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


#Loading dataset
nba_df = pd.read_csv("../input/playoff_shots.csv")
nba_df.head()


# # **EDA**

# In[ ]:


#define correlation of statistics
corr = nba_df.corr()
#create heatmap
plt.subplots(figsize=(15,10))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# This heatmap map shows very l**ittle correlation and insight**, but this is not suprising given the context of the data.
# 
# The most telling insight is that **shot distance is negatively correlated with the % of being made** (shot_made_flag)

# In[ ]:


#In order to better understand the factors around made shots, we need to better visualize the Data

nba_shot_halfcourt_df = nba_df.query('LOC_Y<400')
#Filter shot data within the halfcourt, anything over 400 is an outlier

sns.lmplot('LOC_X', # Horizontal coordinate of shot
           'LOC_Y', # Vertical coordinate of shot
           col="TEAM_NAME", col_wrap= 4, #Display plot by team
           data=nba_shot_halfcourt_df, # Data source
           fit_reg=False, # Don't fix a regression line
           hue='EVENT_TYPE', legend=True,
           scatter_kws={"s": 12}) # S marker size


# With these shot plots, we can generally see where teams generally shoot. also that teams that played more games
# 
# Also we can see that teams that have played more games (won more playoff series) have more shot data

# # **Clean the Data**

# In[ ]:


#Drop non-numerical data fields that statistically irrelevant or covered in another column
nba_df.drop(['GRID_TYPE', 'PLAYER_NAME', 'TEAM_NAME', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', 'HTM', 'VTM'], inplace=True, axis=1)
nba_df.head()


# In[ ]:


#check for missing values
print(pd.isnull(nba_df).sum())


# In[ ]:


#Quantify Shot_Type
shot_type_mapping = {'3PT Field Goal': 3, '2PT Field Goal': 2}
nba_df['SHOT_TYPE'] = nba_df['SHOT_TYPE'].map(shot_type_mapping)
nba_df['SHOT_TYPE'].head(5)


# In[ ]:


#Quantify Shot Zone Range
shot_zone_range_mapping = {'24+ ft.': 24, 'Less Than 8 ft.': 7, '16-24 ft.': 16, '8-16 ft.': 8, 'Back Court Shot': 50}
nba_df['SHOT_ZONE_RANGE'] = nba_df['SHOT_ZONE_RANGE'].map(shot_zone_range_mapping)
nba_df['SHOT_ZONE_RANGE'].head(5)


# In[ ]:


#Quantify Shot Zone Area
shot_zone_area_mapping = {'Back Court(BC)': 0, 'Left Side(L)': 1, 'Left Side Center(LC)': 2, 'Center(C)': 3, 'Right Side Center(RC)': 4, 'Right Side(R)': 5}
nba_df['SHOT_ZONE_AREA'] = nba_df['SHOT_ZONE_AREA'].map(shot_zone_area_mapping)
nba_df['SHOT_ZONE_AREA'].head(5)


# In[ ]:


#Quantify Shot Zone Basic
shot_zone_basic_mapping = {'Backcourt': 0, 'Left Corner 3': 1,'Right Corner 3': 2, 'Above the Break 3': 3, 'Mid-Range': 4, 'In The Paint (Non-RA)': 5, 'Restricted Area': 6}
nba_df['SHOT_ZONE_BASIC'] = nba_df['SHOT_ZONE_BASIC'].map(shot_zone_basic_mapping)
nba_df['SHOT_ZONE_BASIC'].head(5)


# In[ ]:


#Create dummy variable for shotype
shot_dummy = pd.get_dummies(nba_df['ACTION_TYPE'])
nba_df = pd.concat([nba_df,shot_dummy], axis = 1)
nba_df.drop(['ACTION_TYPE'], inplace=True, axis=1)
nba_df.head()


# # **Model**

# In[ ]:


#Split data to predict if the shot was made or missed
X = nba_df.drop('EVENT_TYPE', axis = 1)
y = nba_df['EVENT_TYPE']

#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# In[ ]:


#Predict through Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=350)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
# Check results
print(classification_report(y_test, pred_rfc))


# # **Conclusion**
# 
# Using a simple Random Forest, we are able to predict whether a shot would be made at **over 60%**.
# 
# While this figure is not extremely high, it could serve as an elementary block for NBA analytics.
# 
# Another interesting insight was to see that the model had a much **higher recall rate** (aka True positive Rate) for a **missed shot over a made shot**. This means that a made shot prediction had a **high number of false negatives** (shots that were predicted to miss but actually went in). This might mean that NBA players in the playoffs make shots with low probabilities. 

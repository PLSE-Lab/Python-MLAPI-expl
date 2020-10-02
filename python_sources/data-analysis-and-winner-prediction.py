#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


init_notebook_mode(connected=True) 


# Read input files

# In[ ]:


df_meet = pd.read_csv('../input/meets.csv')
df_open = pd.read_csv('../input/openpowerlifting.csv')


# **Cleaning the Data**
# 
# First, lets look at the empty data on df_meet.

# In[ ]:


sns.heatmap(df_meet.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# The yellow bars represent NaN data on the file. I will remove the MeetTown column, since it will not be used. The MeetState will be used, but the NaN will not affect the results.

# In[ ]:


df_meet.drop('MeetTown',axis=1,inplace=True)


# Now the same with the df_open DataFrame:

# In[ ]:


sns.heatmap(df_open.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# There is a lot of missing data. It would be interesting to analyse the relation between age and other features, but since  a lot of information is missing, I will not consider it here. Moreover, the Squat4Kg, Bench4Kg and Deadlift4Kg are almost all empty. Let's remove those too.  I will also remove the lines without data of the other columns. This dataset is huge, so just eliminating some points should have minor effects.

# In[ ]:


df_open.drop(labels=['Squat4Kg','Bench4Kg','Deadlift4Kg','Age'],axis=1,inplace=True)
df_open.dropna(inplace=True)


# In[ ]:


df_open.info()


# I need the 'Place' column as integer:

# In[ ]:


df_open = df_open[(df_open['Place']!='DQ') & (df_open['Place']!='G')]
df_open['Place'] = df_open['Place'].apply(lambda x: int(x))
df_open.info()


# **Visualyzing the data:**
# 
# Relation between Males and females

# In[ ]:


sns.countplot(x='Sex', data=df_open)


# In[ ]:


sns.pairplot(df_open, vars=['BodyweightKg','BestSquatKg','BestBenchKg',
                            'BestDeadliftKg'],hue='Sex',palette='coolwarm')


# The histograms show that the women average total weight are smaller than those for men, which makes sense.
# There is something strange in the BestSquatKg column. There are some negative values. This doesn't make sense. Since just two points are negative, I will just remove those.

# In[ ]:


df_open = df_open[df_open['BestSquatKg']>0]


# In[ ]:


#sns.pairplot(df_open, 
#             vars=['BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg'],hue='Sex',palette='coolwarm')


# **Distribution of the meets in the US states**

# In[ ]:


#Select meets from the US
df_usa = df_meet[df_meet['MeetCountry']=='USA']
#Organize and counts meets by state
df_state = df_usa.groupby('MeetState').count()
df_state.reset_index(inplace=True)


# In[ ]:


#Data and layout dictionaries for plotly
data = dict(type='choropleth',
            colorscale = 'Viridis',
            locations = df_state['MeetState'],
            z = df_state['MeetID'],
            locationmode = 'USA-states',
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
            colorbar = {'title':"Number of Meets"}
            )
layout = dict(title = 'Number of meets by State',
              geo = dict(scope='usa',
                         showlakes = False,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)


# California and Texas had the biggest number of Meets!

# **Winner prediction**
# 
# Can we predict the winner based on the weight and sex, and best performance on Squat, Deadlift and Benchpress?
# I will use the random forest algorithm.
# 

# In[ ]:


#Creating another DataFrame, that will be used in the predictions
df_prov = pd.DataFrame.copy(df_open)
#Removing object columns that will not be used
df_prov.drop(labels=['MeetID','Name','Division','WeightClassKg','Equipment','Wilks'],axis=1,inplace=True)


# In[ ]:


#Categorize Sex column
cat_feats = ['Sex']
df_predict = pd.get_dummies(df_prov,columns=cat_feats,drop_first=True)


# In[ ]:


#Functions that returns 1 if the person is the winner (Place == 1) and 0 otherwise
def change_place(val):
    if val > 1:
        return 0
    else:
        return 1


# In[ ]:


df_predict['Place']=df_predict['Place'].apply(change_place)
df_predict.head()


# In[ ]:


#Importing Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df_predict.drop('Place',axis=1), df_predict['Place'], 
                                                    test_size=0.3)


# In[ ]:


rfc = RandomForestClassifier()


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


pred_rfc = rfc.predict(X_test)


# In[ ]:


print('Classification report for Random Forest:')
print(classification_report(y_test,pred_rfc))


# At first, I thought the accuracy was bad. Since I know the body weight, sex, and best performance, I should be able to predict if the person would win or not. However, these are results for meets in the whole world. A result sufficient to win in one place might not be enought in another country, for instance. Moreover, the information about the division is not used, so we might be mixing amateur with professionals, among other things. This can be visualize with a boxplot of the TotalKg:

# In[ ]:


sns.boxplot(x='Place',y='TotalKg',data=df_predict)


# The means for Place==1 is slightly larger, as expected, but the deviations are large. Thus, it might be difficult for the algorithm to predict the result correctly. With this in mind, a 60% accuracy is not bad!  

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import os
warnings.filterwarnings('ignore')
os.listdir('../input/2018-chicago-crime-data')


# ## Getting Started
# Import the dataset with Pandas and check if nulls are going to be an issue.

# In[ ]:


df = pd.read_csv('../input/2018-chicago-crime-data/Crimes_-_2018.csv')
print(df.isnull().sum())
print("--------------------------")
print("this dataset has ",len(df)," observations")


# #### Delete Some Nulls
# Looks like nulls are going to be a problem. I want to do most of my analysis around Lon/Lat so I'm going to have to clean this up. Seeing as this is a massive dataset, I'm not concerted with deleting 4,600 rows. We should stil, have enough data for our model.

# In[ ]:


df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(2)


# ## Build the Model
# ### First prep the Data
# Let's scale the data to make is more digestable for the model using StandardScaler. Let's try to create model that tries to predict what type of crime has occered based on Longatude and Latitude.

# In[ ]:


from sklearn.preprocessing import StandardScaler
sf = df[['Primary Type','Longitude','Latitude']]
scaler = StandardScaler()
scaler.fit(sf.drop('Primary Type',axis=1))
scaled_features = scaler.transform(sf.drop('Primary Type',axis=1))
sf_feat = pd.DataFrame(scaled_features,columns=sf.columns[1:])
sf_feat.head()


# ### Format the data into our Train - Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,sf['Primary Type'],
                                                    test_size=0.30)


# ### Run our first K-Nearest Neighbors Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))


# ### Results
# The results show _very_ different results for different types of crimes. This shows us that some crimes tend to take place in similar locations, wheras others are more random in nature. Lets graph this to get a better idea of what this loks like.

# In[ ]:


results = pd.DataFrame(classification_report(y_test,pred,output_dict=True))
results = results.swapaxes("index", "columns") 
results['categories'] = results.index
results = results.sort_values('f1-score',ascending=0)
results.drop(['accuracy','macro avg','weighted avg'],inplace=True)
results.insert(0,'K-Value','k=1')
results.reset_index(drop=True, inplace=True)
fig = px.bar(results,x='categories',y="f1-score",color_discrete_sequence=('#00A8E8','#003459'),
             opacity=.7,title='F1 Scores by Crime Type')
fig.show()


# ### Optimize the Model
# There are a few different ways to optimize a K-Neighbors Model:
# - Changing the K value (n_neighbors = [])
# - Changing the Distance Function (p = [1,2])
# - Change you variables
# 
# For this model I'm going to look for our optimal K-value. Fair warning, this takes some time to run

# In[ ]:


error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
error_rt = pd.DataFrame(error_rate,columns = ['error rate'])
error_rt['K-value'] = range(1,40)
fig = px.line(error_rt,x='K-value',y='error rate',color_discrete_sequence=('#1D3557','#00A8E8')
             ,title='Error Rate Using Different K-Values')
fig.show()


# ### Change the K-value and re-run
# We can slightly decrease our error rate by increasing our K-value to 25. Lets do this and see how this alters our results.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
optimized_results = pd.DataFrame(classification_report(y_test,pred,output_dict=True))
optimized_results = optimized_results.swapaxes("index", "columns") 
optimized_results['categories'] = optimized_results.index
optimized_results.drop(['accuracy','macro avg','weighted avg'],inplace=True)
optimized_results.insert(0,'K-Value','k=25')
optimized_results.reset_index(drop=True, inplace=True)
combined = results.append(optimized_results,ignore_index = True)
combined = combined.sort_values('f1-score',ascending=0)
fig = px.bar(combined,x='categories',y="f1-score",color='K-Value',barmode='group',color_discrete_sequence=('#003459','#00A8E8'),
             opacity=.7,title='F1 Scores Using Different K-Values')
fig.show()


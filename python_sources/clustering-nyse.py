#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%reset -f                       # Reset memory

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


# In[ ]:


np.set_printoptions(precision = 3,          # Display upto 3 decimal places
                    threshold=np.inf        # Display full array
                    )


# In[ ]:


# Display output of command in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
#pd.options.display.float_format = '${:,.2f}'.format
pd.options.display.float_format = '{:.2f}'.format


# In[ ]:


# Go to folder containing data file
#os.chdir("E:\\IDA_Training\\nyse_Exercise\\854_1575_bundle_archive")
#os.listdir()            # List all files in the folder

# 2.1 Read file and while reading file,
#      convert 'Timestamp' to datetime time
#df = pd.read_csv("fundamentals.csv",
 #                 parse_dates = ['Period Ending']    # especial for date parsing
#                  )

df = pd.read_csv("../input/nyse/fundamentals.csv",
                  parse_dates = ['Period Ending']    # especial for date parsing
                  )
df.head()


# In[ ]:


df.shape               
df.dtypes


# In[ ]:


df.columns = df.columns.str.replace(".", "")
df.columns = df.columns.str.replace(" / ", "_")
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(",", "")
df.columns = df.columns.str.replace("-", "")
df.columns = df.columns.str.replace("'", "")
df.columns = df.columns.str.replace("&", "")


# In[ ]:


df.shape
df.dropna(axis=1,inplace=True)
df.shape


# In[ ]:


df.head()


# In[ ]:


df=df.groupby(['Ticker_Symbol']).mean()
df =df.reset_index()
df.head()


# In[ ]:


#plt.figure(figsize=(10, 8))
top10 = df.nlargest(10, "Gross_Profit")
sns.barplot(x="Ticker_Symbol", y="Gross_Profit", data=top10)


# In[ ]:


#if Net profit > 300 crores will yeild divident
df['Net_profit']=df['Gross_Profit']-df['Income_Tax']
df['Divident_yeild'] = [1 if x >3000000000 else 0 for x in df['Net_profit']]
gr=df.groupby(['Divident_yeild'])
gr['Divident_yeild'].agg(['count'])


# In[ ]:


sns.distplot(df.Net_profit) 


# In[ ]:


sns.catplot('Divident_yeild','Net_profit', data = df, kind = 'box') 


# In[ ]:


#df.head()
#top20 = df.nlargest(20, "Net_profit")
top20 = df.nlargest(20, 'Net_profit')
bottom20 = df.nsmallest(20, 'Net_profit')
frames = [top20, bottom20]
result = pd.concat(frames)
sns.catplot('Divident_yeild','Net_profit', data = result, kind = 'box')


# In[ ]:


y = df['Divident_yeild'].values
df.drop(columns = ['Divident_yeild'], inplace = True)


# In[ ]:


num_columns = df.select_dtypes(include = ['float64', 'int64']).copy()

ss = StandardScaler()     # Create an instance of class
ss.fit(num_columns)                # Train object on the data
X = ss.transform(num_columns)      # Transform data
X[:2, :]                  # See first 2 rows


# In[ ]:


X_train, X_test, _, y_test = train_test_split( X,               
                                               y,               
                                               test_size = 0.25 
                                               )

X_train.shape              
X_test.shape               


# In[ ]:


clf = KMeans(n_clusters = 2)
clf.fit(X_train)
clf.cluster_centers_
clf.cluster_centers_.shape         
clf.labels_                        
clf.labels_.size                   
clf.inertia_                       
silhouette_score(X_train, clf.labels_)    


# In[ ]:


y_pred = clf.predict(X_test)
y_pred

np.sum(y_pred == y_test)/y_test.size

dx = pd.Series(X_test[:, 0])
dy = pd.Series(X_test[:,1])
sns.scatterplot(dx,dy, hue = y_pred)


# In[ ]:


#  Scree plot:
sse = []
for i,j in enumerate(range(10)):
      n_clusters = i+1
 
    clf1 = KMeans(n_clusters = n_clusters)
    
    clf1.fit(X_train)
  
    sse.append(clf1.inertia_ )

# Plot the line now
sns.lineplot(range(1, 11), sse)


# In[ ]:


visualizer = SilhouetteVisualizer(clf, colors='yellowbrick')
visualizer.fit(X_train)        
visualizer.show()              


# In[ ]:


# Intercluster distance: Does not work
from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(clf)
visualizer.fit(X_train)        
visualizer.show()              


# In[ ]:





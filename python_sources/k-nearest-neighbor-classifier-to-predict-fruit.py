#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_table('../input/fruit_data_with_colors.txt')


# First of we need to remove all the string values from the data and for that we need to find the relation between fruit_subtype and fruit_label.
# 
# For that what I have done is made a bar plot having 'fruit_subtype' on X- Axis and 'fruit_label' on Y -Axis.

# In[ ]:



plt.figure(figsize=(15,7))
sns.barplot('fruit_subtype','fruit_label',data=df)
plt.show()


# Looking at the above bar chart, we need to categorize our fruit_subtype into their respective sub_categories 
# 
# Like  : granny_smith,braeburn,golden_delicious,cripps_pink are some fruit_subtypes which fall in fruit_label 1 ,therefore we need to map the fruit_types to their subcategories.

# In[ ]:


fruit_type={'turkey_navel' : 3 ,'unknown' : 4, 'cripps_pink' : 1
            ,'selected_seconds' : 3, 'spanish_belsan' : 4, 'golden_delicious' : 1,
            'braeburn' : 1, 'mandarin' : 2, 'spanish_jumbo' : 3,'granny_smith' : 1}

df['fruit_subtype']=df['fruit_subtype'].map(fruit_type)


# Can we make a new column by observing fruit's height?...well I think answer is no..
# 
# What i did was grouping all the heights of fruits by the fruit_label and the results were almost similar,same  was the approach with width,mass and colour_score but it failed!...

# In[ ]:


df['height'].groupby(df['fruit_label']).mean()


# In[ ]:


df['width'].groupby(df['fruit_label']).mean()


# In[ ]:


df['mass'].groupby(df['fruit_label']).mean()


# In[ ]:


df['color_score'].groupby(df['fruit_label']).mean()


# In[ ]:


from sklearn.model_selection import train_test_split
cols=['fruit_subtype', 'mass', 'width', 'height','color_score']
X=df[cols]

y=df['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# Fun Part Begins!...let's find which is the most appropriate value of our K,
# 
# I have taken a range from 1-9 so let's see for which value of K we get highest score!

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

neighbours=np.arange(1,9)
train_accuracy=np.empty(len(neighbours))
test_accuracy=np.empty(len(neighbours))


# In[ ]:


for i in range(len(neighbours)):
    knn=KNeighborsClassifier(n_neighbors=i+1)
    knn.fit(X_train,y_train)
    train_accuracy[i]=knn.score(X_train,y_train)
    test_accuracy[i]=knn.score(X_test,y_test)


# In[ ]:


plt.title('k-NN Varying number of neighbors')
plt.plot(neighbours, test_accuracy, label='Testing Accuracy')
plt.plot(neighbours, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


# So by looking at the above graph, we can say that value of K can either be 6 or 4,let's take 6!

# In[ ]:



knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# In[ ]:





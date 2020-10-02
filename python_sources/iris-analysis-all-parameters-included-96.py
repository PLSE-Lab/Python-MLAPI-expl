#!/usr/bin/env python
# coding: utf-8

# # Iris Species Analysis

# Let's import our libraries and take a look at the data.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

dataset.head()


# There's not much to work with. Let's take a closer look at each individual feature in relation to the species.

# In[ ]:


dataset.describe()


# In[ ]:


dataset['species'].value_counts()


# In[ ]:


fig = dataset[dataset['species'] == 'Iris-setosa'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='orange', label='Setosa')
dataset[dataset['species'] == 'Iris-versicolor'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='yellow', label='Versicolor', ax=fig)
dataset[dataset['species'] == 'Iris-virginica'].plot(kind='Scatter', x='sepal_length', y='sepal_width', color='blue', label='Verginica', ax=fig)
fig.set_ylabel('Sepal Width')
fig.set_xlabel('Sepal Length')
fig.set_title('Sepal Length vs Width')

fig = plt.gcf()
fig.set_size_inches(18, 9)
plt.show()


# The plot above tells us a bit about the indication of sepal length and width in relation to the species. Let's take a look at how the `petal_width` and `petal_length` affect the determination of the species.

# In[ ]:


dataset['species'].groupby(dataset['petal_width']).value_counts()


# At face value this doesn't seem to be very clear. We can only conclusively tell that Iris Setosa is distinct in terms of it's petal dimensions though. This is the same conclusion as the sepal dimensions that we've seen from the scatter plot above.

# In[ ]:


dataset['species'].groupby(pd.qcut(dataset['petal_width'], 3)).value_counts()


# Since we now know that the petal and sepal dimensions play a big role in the species, we now know to quantify this feature into a new column. We can create `sepal_area` and `petal_area` columns respectively.

# In[ ]:


dataset['petal_area'] = dataset.apply(lambda row: (row['petal_length'] * row['petal_width']), axis=1)


# In[ ]:


dataset.head()


# In[ ]:


dataset['species'].groupby(pd.qcut(dataset['petal_area'], 3)).value_counts()


# In[ ]:


dataset['sepal_area'] = dataset.apply(lambda row: (row['sepal_length'] * row['sepal_width']), axis=1)


# In[ ]:


dataset['lensq'] = dataset.apply(lambda row: (row['sepal_length'] * row['sepal_width']), axis=1)


# In[ ]:


dataset['widsq'] = dataset.apply(lambda row: (row['petal_width'] * row['petal_length']), axis=1)


# In[ ]:


dataset['squares'] = dataset.apply(lambda row: (row['lensq'] - row['widsq']), axis=1)


# In[ ]:


dataset.head()


# In[ ]:


dataset['species'].groupby(pd.qcut(dataset['sepal_area'], 3)).value_counts()


# Interestingly, the `sepal_area` column is a bit more varied than the `petal_area` column. We can also include another feature here. From our two new columns, we can make a third column from the difference between them called `area_diff`. This may give us more insight. Before doing that though, we can plot the `petal_area` with `sepal_area` to see if any linear relationship exists between the two columns.

# In[ ]:


sns.lineplot(dataset['petal_area'], dataset['sepal_area'])


# No clear correlation found between sepal_area and petal_area. There is a weak correlation found if the species 'setosa' is removed from the dataset.

# In[ ]:


dataset['area_diff'] = dataset.apply(lambda row: (row['sepal_area'] - row['petal_area']), axis=1)


# In[ ]:


dataset.head()


# In[ ]:


dataset['species'].groupby(pd.qcut(dataset['area_diff'], 3)).value_counts()


# In[ ]:


columnsTitles = ['sepal_length', 'sepal_width', 'petal_length','petal_width','petal_area', 'sepal_area','area_diff','lensq','widsq','squares','species']
dataset=dataset.reindex(columns=columnsTitles)


# In[ ]:


dataset.head(3)


# We can finally move onto building the model. Our classifier uses only the columns that we've engineered.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(dataset.iloc[:, 0:-2], dataset.iloc[:, -1])
print("%.4f" % rf.oob_score_)


# from sklearn.ensemble import RandomForestClassifier
# 
# rf = RandomForestClassifier(criterion='gini', 
#                              n_estimators=1000,
#                              min_samples_split=10,
#                              min_samples_leaf=1,
#                              max_features='auto',
#                              oob_score=True,
#                              random_state=1,
#                              n_jobs=-1)
# rf.fit(dataset['area_diff'], dataset['species'])
# print("%.4f" % rf.oob_score_)

# It gives us an 'out of bag' score of 95.33%. It is possible to acheive 100% with the SVM model but a clear difference can only be shown with a larger dataset. Still, this is a very respectable score.

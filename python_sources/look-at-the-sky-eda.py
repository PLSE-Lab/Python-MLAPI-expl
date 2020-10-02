#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')


# In[ ]:


train_df.head(10)


# In[ ]:


train_df = train_df[~train_df['EncodedPixels'].isnull()]
train_df['Image'] = train_df['Image_Label'].map(lambda x: x.split('_')[0])
train_df['Class'] = train_df['Image_Label'].map(lambda x: x.split('_')[1])
classes = train_df['Class'].unique()
train_df = train_df.groupby('Image')['Class'].agg(set).reset_index()

count_of_class = [] 
for class_name in classes:
    train_df[class_name] = train_df['Class'].map(lambda x: 1 if class_name in x else 0)
    count_of_class.append(train_df[train_df[class_name] == 1][class_name].value_counts()[1])

count_of_class = pd.DataFrame({
    'Class': classes,
    'Count': count_of_class
}) 
      
train_df['Exclusive'] = train_df['Class'].map(lambda x: 1 if len(x) == 1 else 0)
train_df['Count_Class_Overlap'] = train_df['Fish'] + train_df['Flower'] + train_df['Sugar'] + train_df['Gravel']
train_df['Class_str'] = train_df['Class'].astype(str)

train_df.head(10)


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(6,7)
sns.countplot(x='Exclusive', data=train_df, palette="Blues_d", ax=ax)
ax.set_title('Multiple Classification vs Single Classification')
ax.set_xticklabels(['Multiple Cassification','Single Classification'])
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1,
        train_df['Exclusive'].value_counts()[i],ha="center")
    i += 1


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10,4)
sns.barplot(x='Class', y='Count', data=count_of_class.sort_values(by='Count', ascending=False), palette="Blues_d", ax=ax)
ax.set_title('Most Commom Class')
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1,
        count_of_class.sort_values(by='Count', ascending=False)['Count'][i],ha="center")
    i += 1


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10,4)
sns.barplot(x='Count_Class_Overlap', y='Image', data=train_df.groupby('Count_Class_Overlap').count()[['Image']].reset_index(), palette="Blues_d", ax=ax)
ax.set_title('Images with multiple classification')
ax.set_xticklabels(['One Class','Two Classes', 'Three Classes', 'Four Classes'])
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1,
        train_df.groupby('Count_Class_Overlap').count()[['Image']].iloc[i]['Image'],ha="center")
    i += 1


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16,8)
sns.barplot(
    y='Class_str', 
    x='Image', 
    data=train_df[train_df['Exclusive'] == 0].groupby('Class_str').count()[['Image']].sort_values(by='Image', ascending=False).reset_index(), palette="Blues_d", ax=ax)
ax.set_title('Most Common Combination of Class')
i=0
for p in ax.patches:
    width = p.get_width()
    ax.text(width + 20, p.get_y()+p.get_height()/2.,
        train_df[train_df['Exclusive'] == 0].groupby('Class_str').count()[['Image']].sort_values(by='Image', ascending=False).iloc[i]['Image'],ha="center")
    i += 1


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(10,4)
sns.barplot(
    y='Image', 
    x='Class_str', 
    data=train_df[train_df['Exclusive'] == 1].groupby('Class_str').count()[['Image']].sort_values(by='Image', ascending=False).reset_index(), palette="Blues_d", ax=ax)
ax.set_title('Most Common Single Classification Class')
i=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height + 0.1,
        train_df[train_df['Exclusive'] == 1].groupby('Class_str').count()[['Image']].sort_values(by='Image', ascending=False).iloc[i]['Image'],ha="center")
    i += 1


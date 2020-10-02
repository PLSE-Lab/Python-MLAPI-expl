#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Data Preprocessing

# In[ ]:


df.isnull().any()


# #### salary is only null coulmns
# #### lets see whether salary is null for those who were not placed or for others too

# In[ ]:


df['status'].unique()


# In[ ]:


df[df['status'] == 'Not Placed']['salary'].unique() # column has only null values


# hence those who were not places have the salary set null in the data

# lets fill them with 0

# In[ ]:


df['salary'].fillna(0.0, inplace = True)


# In[ ]:


df.isnull().any()


# #### Now, no one has null values in the any column 

# In[ ]:


df.head()


# #### Adding an additional column 'Mean Score' to describe score of all the education level

# In[ ]:


df['Mean Score'] = df['ssc_p'] + df ['hsc_p'] + df['degree_p'] +df['mba_p']
df['Mean Score'] = df['Mean Score'] / 4


# # Data Visualization

# In[ ]:


sns.distplot(df['salary'], kde = False)
plt.xlabel('salary')
plt.ylabel('number of students')
plt.title('Number of students and salary range')


# #### Clearly most of the placed student are in the range of 2 Lac to 3 Lac

# ## Placement visualization

# In[ ]:


placed = df[df['status'] == 'Placed']
unplaced =df[df['status'] == 'Not Placed']


# ### 1. Placed and unplaced

# In[ ]:


sns.catplot(x="status", kind="count", data=df);


# ### 2. Based on gender

# In[ ]:


sns.catplot(x="status", kind="count",hue ='gender', data=df);


# ### Inference: The performance of Females in placement is poor as ratio of placed to unplaced males is greater than in comparison with females

# In[ ]:


ax = plt.subplot(111)
sns.scatterplot(x='Mean Score',y='salary',hue='gender',data= placed)
ax.legend(bbox_to_anchor=(1.3, 1.0))


# ### Inference: There are more number of males in higher slab of salary than females

# In[ ]:


sns.catplot(x="workex", kind="count",hue ='gender', data=df, col='status');


# ### Inference: 
#     There are few males and females both who has prior work experience and were not placed. Thus working experience increases the placement chances irrespective of your Gender.

# ## 3. Based on specialisation

# In[ ]:


sns.catplot(x="specialisation", kind="count", data=df, col='status');


# In[ ]:


spec = np.asarray(df['specialisation'].unique())
placedSpec = list(map(lambda spec: len(placed[placed['specialisation'] == spec]), spec))
plt.pie(x = placedSpec, shadow = True , labels = spec, radius = 1.5, startangle=90)
plt.title('Placed')
plt.show()
unplacedSpec = list(map(lambda spec: len(unplaced[unplaced['specialisation'] == spec]), spec))
plt.pie(x = unplacedSpec, shadow = True , labels = spec, radius = 1.5, startangle=90)
plt.title('Not placed')
plt.show()


# ### Inference: 
#     Students of Mkt&Fin gets slight more placement than students of Mkt&Hr

# In[ ]:


ax = plt.subplot(111)
sns.scatterplot(x='Mean Score',y='salary',hue='specialisation',data= placed)
ax.legend(bbox_to_anchor=(1.4, 1.0))


# ### Inference: 
#     1. Most of the students of higher slab of salry are from Mkt&Fin .
#     2. The average performance throughout career of Mkt&Fin is better than Mkt&HR.

# In[ ]:


placed.head()


# ## 4. Based on undergrad degree

# In[ ]:


sns.catplot(x="degree_t", kind="count", data=df, col ='status');


# ### Inference: 
#     1. Students of Comm & Mgmt gets almost double number of placement than rest of other courses combined.
#     2. The ratio of placed and unplaced students of Comm&Mgmt & Sci&Tech are same. Hence chances of both type of students getting placed is almost same.

# In[ ]:


degree = np.asarray(df['degree_t'].unique())
placedDegree = list(map(lambda deg: len(placed[placed['degree_t'] == deg]), degree))
plt.pie(x = placedDegree, shadow = True , labels = degree, radius = 1.5 )
plt.title('Placed')
plt.show()

unplacedDegree = list(map(lambda deg: len(unplaced[unplaced['degree_t'] == deg]), degree))
plt.pie(x = unplacedDegree, shadow = True , labels = degree, radius = 1.5 )
plt.title('Not placed')
plt.show()


# ### Inference: 
#     1. The level of similarity of both the pie charts shows that being from sci&Tech or Comm&Mgmt wont effect your placement probabilty.

# In[ ]:


ax = plt.subplot(111)
sns.scatterplot(x='Mean Score',y='salary',hue='degree_t',data= placed)
ax.legend(bbox_to_anchor=(1.4, 1.0))


# ### Inference: 
#     1. Interestingly, more Sci&Tech students grabed salary more than 5Lac than Comm&Mgmt students.
#     2. But the number start becoming uniform as the slab comes lower.

# ## 5. Based on Employability test percentage

# In[ ]:


sns.jointplot(x='etest_p', y = 'salary', data = placed , kind="hex", color="#4CB391")


# ### Inference: 
#     Having high scores in empployement test wont result to the higher salary as the placed are almost uniformly distributed throughout hte score and band width of for most of the students is between 2 Lacs to 3 Lacs

# In[ ]:


sns.jointplot(x='Mean Score', y = 'etest_p', data = placed , kind="hex", color="#4CB391")


# ### Inference: 
#     The plot shows that etest_p does not  creates much differenec wrt to students overall academic performance.

# In[ ]:


sns.jointplot(x='degree_p', y = 'salary', data = placed , kind="hex", color="#4CB391")


# In[ ]:


sns.jointplot(x='Mean Score', y = 'salary', data = placed , kind="hex", color="#4CB391")


# ### Inference: 
#     Student with average score throughout the career perform better than high scored student

# ## 6. Based on Experience

# In[ ]:


df.head()


# In[ ]:


sns.catplot(x="workex", kind="count", data=df, col ='status');


# ### Inference: 
#     Having a prior experinece increases the chances of placement.

# In[ ]:


sns.catplot(x="workex", kind="count", data=df, hue = 'gender');


# ### Inference: 
#     1. There more numbers of unexperinced studnets than experineced.
#     2. There more experinecd males than females. 

# ## 7. Correlation

# In[ ]:


df.head()


# In[ ]:


scoresMatPlaced = placed.loc[:, ['ssc_p','hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary', 'Mean Score']]


# In[ ]:


scoresMatPlaced.head()


# In[ ]:


sns.heatmap(scoresMatPlaced.corr(), cmap = 'coolwarm')


# In[ ]:


sns.clustermap(scoresMatPlaced.corr(), cmap ='coolwarm')


# ### Inference: 
#     No exams throughout the career is correlated with salary.

# # Thank You

# In[ ]:





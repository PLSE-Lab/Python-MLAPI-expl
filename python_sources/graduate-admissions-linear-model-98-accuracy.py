#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[ ]:


dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')


# In[ ]:


dataset.columns


# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# No missing data, let's move on

#  ### Describing the data
#  - 1. GRE Scores ( out of 340 ) 
#  - 2. TOEFL Scores ( out of 120 ) 
#  - 3. University Rating ( out of 5 ) 
#  - 4. Statement of Purpose (SOP) and Letter of Recommendation Strength ( out of 5 ) 
#  - 5. Undergraduate GPA ( out of 10 ) 
#  - 6. Research Experience ( either 0 or 1 )
#  - 7. Chance of Admit ( ranging from 0 to 1 )

# In[ ]:


dataset.describe()


# ## After analysing the data above, I may conclude that: 
# - The GRE of the students is elevated because the STD shows that its variation is about of 11.47 points, so most of all they are all at the same range, which for me is extremilly high, since it is out of 340. 
# - The same occurred with the TOEFL Score, they are all around the same score.

# In[ ]:


plt.figure(1, figsize=(10,6))
plt.subplot(1,3, 1)
plt.boxplot(dataset['GRE Score'])
plt.title('GRE Score')

plt.subplot(1,3,2)
plt.boxplot(dataset['TOEFL Score'])
plt.title('TOEFL Score')

plt.subplot(1,3,3)
plt.boxplot(dataset['University Rating'])
plt.title('University Rating')

plt.show()


# ### As we can see there is no outlier, which means that the data is well distributed, in other words, all the students got the same score. This confirms our hypotesis described above
# 
# Now let's check if the best students want the best Universities

# In[ ]:


university_rating = dataset.groupby('University Rating')['GRE Score'].mean()


# In[ ]:


plt.bar(university_rating.index, university_rating.values)
plt.title('University Rating X GRE Score')
plt.ylabel('GRE Score')
plt.xlabel('University Rating')
plt.show()


# We can conclude that the top students are at the top Universities, the ones with the highest score and the ones with the second highest score; however, the difference is not that big amount the universities, even though the STD of the scores was not too big to create this type of scenario

# In[ ]:


pd.DataFrame(dataset.corr()['Chance of Admit '])


# With the correlation is possible to realize that, the students that have the highest chance to be admited in a University is the ones that have decent Undergraduate GPA (CGPA), GRE Score and TOEFL Score. I also thought that the ones with an research project could also have a better chance, but this assumption was not right. Since Serial No. is only a sequential number, it will not improve our model, so I will remove it.

# In[ ]:


dataset.drop(columns=['Serial No.'], axis=1, inplace=True)


# ## QUESTION
# - Does the students whose have a high GRE Score does also have a high TOEFL Score?

# In[ ]:


plt.figure(figsize=(10,6))
plt.scatter(dataset['GRE Score'], dataset['TOEFL Score'])
plt.title('GRE Score X TOEFL Score')
plt.xlabel('GRE Score')
plt.ylabel('TOEFL Score')
plt.show()


# With the scatter plot we can see that the students whose have a great score also have a good TOEFL score and also the students that does not have a great GRE Score they do not have a good TOEFL Score

# # PreProcessing
# It always seems to be nice to normalize the data, so let's do it

# In[ ]:


dataset['GRE Score'] = preprocessing.StandardScaler().fit_transform(dataset['GRE Score'].values.reshape(-1,1))
dataset['TOEFL Score'] = preprocessing.StandardScaler().fit_transform(dataset['TOEFL Score'].values.reshape(-1,1))


# ## Why does not use onehotencoder in the Research feature?
# The research feature is what allow us to have a ideia whether or not the student did a research project. Since make sense to give a better grade to sudents that have a project than the ones that does not have. I am not going to use OneHotEncoder on this variable.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset.iloc[:,0:], dataset.iloc[:,-1], random_state=42)


# In[ ]:


from sklearn import linear_model
lr = linear_model.Ridge(alpha=0.5)
lr.fit(X_train, y_train)


# In[ ]:


lr.score(X_test,y_test)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)


# In[ ]:


rfr.score(X_test,y_test)


# # Results
# With this data we did not have to fill any null data, we only need to normalize the data, since the scale amoung the values are different, we need to normalize the data. The prediction was pretty much high.

#!/usr/bin/env python
# coding: utf-8

# Machine Leaning models on HR analytics data:

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
hr = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')
hr.head()


# checking of any null values in data

# In[ ]:


hr.isnull().sum()


# Histogram with satisfaction levels

# In[ ]:


plt.hist(hr.satisfaction_level)


# % of "less satisfaction level, more average hours woked and ressigned"

# In[ ]:


hr_uns = hr[(hr['average_montly_hours'] > 220) & (hr['satisfaction_level'] < 0.5) & (hr.left == 1)]
hr_uns.head()
hr_uns_p = round((len(hr_uns)/len(hr))*100, 2) 
hr_uns_p


# Department wise resigned detils: 

# In[ ]:


hr.groupby(['Department','left']).size().unstack('left').plot(kind='bar')


# With the below plot we can assume even more avg. monthly hours and good last evaluation but left due to low salary.

# In[ ]:


b=hr[(hr['last_evaluation']>0.7) & (hr['average_montly_hours']>220) & hr['left']==1]
sns.countplot(data=b, x=b.salary, palette= 'autumn', order =['low', 'medium', 'high'])


# In[ ]:


sns.countplot(data = hr, x = 'salary', hue ='left',palette = 'bwr')


# changing salary data to numarical 

# In[ ]:


salary = pd.get_dummies(hr.salary, drop_first = True)
hr1 = pd.concat([hr,salary], axis=1)
hr1.head()


# dropping unnecessary columns 

# In[ ]:


hr1 = hr1.drop(['Department','salary'], axis=1)
hr1.head()


# creating train and test data from dataframe

# In[ ]:


x = hr1.drop(['left'], axis=1)
y = hr1.left

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1 )


# LogisticRegression model:

# In[ ]:



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
model.score(x_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# the LogisticRegression model given 78% accuracy score.

# DecisionTree model:

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model1=  DecisionTreeClassifier()
model1.fit(x_train, y_train)
y_pred1 = model1.predict(x_test)
model1.score(x_test, y_test)


# the DecisionTree model given 98.5% accuracy score.

# RandomForest:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model2=  RandomForestClassifier(n_estimators=15)
model2.fit(x_train, y_train)
y_pred1 = model2.predict(x_test)
model2.score(x_test, y_test)


# the RandomForest model given 99.3% accuracy score with 15 trees(n_estimators) .

# support vector machine:

# In[ ]:


from sklearn.svm import SVC
model3=  SVC()
model3.fit(x_train, y_train)
y_pred1 = model3.predict(x_test)
model3.score(x_test, y_test)


# the SVM model given 95.7% accuracy score.

# After checking all models RandomForest given good accuracy of 99.3%

# Feel free to comment with suggestions or any clarifications for further active discussions. 
# 
# e-mailid: mcommahesh@gmail.com
# 

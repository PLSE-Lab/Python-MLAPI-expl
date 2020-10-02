#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read in data, get idea of dimension and column values
humans = pd.read_csv('../input/HR_comma_sep.csv')
humans.sample(n = 15)


# In[ ]:


# some sample statistics
humans.describe()


# In[ ]:


# scatterplot matrix
sb.pairplot(humans, hue = 'left', palette='husl')


# In[ ]:


group_by_project = humans.groupby(['left','number_project'])['satisfaction_level'].aggregate(np.mean)
group_by_project


# As it stands, of those who left, the ones who were taking on too many projects had lower average job satisfaction levels. It would also appear that too few projects factored into leaving. Could it be due to too many work projects and too little pay or not enough of a challenge?  Let's check out the salaries compared against number of projects and look at how many people left. 

# In[ ]:


sb.factorplot('number_project', # x-axis, number of projects
               col='salary', # one plot for each salary type
              data=humans, # data to analyze
              kind = 'count', # kind of plot to be made
              size = 3, aspect = 0.7, # sizing parameters
              hue = 'left') # color code by left/stayed


# From the factor plots above, it appears as though salaries coupled with low work project numbers is a possible reason for leaving. That's...odd. Perhaps it's not the number of projects we should be looking at but the average number of weekly hours. 

# In[ ]:


sb.factorplot('left', # x axis, number of boxplots per graph
              'average_montly_hours', # column to create boxplots from
              data = humans, 
              kind = 'box', # create boxplots
              col='salary', # number of graphs to make
              size = 3, # sizing parameters 
              aspect = 0.8)


# Interestingly enough, there appears to be no difference in the boxplots of the employees who stayed. However, of those who left (for the high salary workers), the median number of hours is significantly lower than that of the medium and low salary workers. Interesting. Let's revisit those satisfaction levels. 

# In[ ]:


sb.factorplot('left',
             'satisfaction_level', 
             data = humans, 
             kind = 'box',
             col = 'salary',
             size = 3, 
             aspect = 0.8)


# Among the high salary workers, the spread of satisfaction levels is very low. Let's see what the satisfaction level looks like overall before nesting into different categories. 

# In[ ]:


sb.boxplot(x='left', y='satisfaction_level', data=humans)


# Ok, so this isn't so surprising, the satisfaction levels of those who left varies greatly while the median of that group is about 0.4. Of those who stuck around, the median is about 0.7, a notable difference. 

# ## Let's create some dummy variables in order to run a random forest classifier. Of course, we'll need these to run logistic regression and make a decision tree. Multitasking! 

# In[ ]:


# encode labels for the sales and salary variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
humans["sales"] = le.fit_transform(humans["sales"])
humans["salary"] = le.fit_transform(humans["salary"])


# In[ ]:


sales_df = pd.get_dummies(humans['sales'], prefix = 'sales')
salary_df = pd.get_dummies(humans['salary'], prefix='sal')
sales_df.head()


# In[ ]:


# concatenate the frames with the original dataframe 
humans = pd.concat([humans,sales_df,salary_df], axis = 1)
humans.sample(n = 15)


# In[ ]:


# create training and test sets
from sklearn import tree 
from sklearn.cross_validation import train_test_split

humans.drop('sales', inplace=True,axis = 1)
humans.drop('salary', inplace = True, axis = 1)
y = humans.pop('left')
X_train, X_test, Y_train, Y_test = train_test_split(humans, y, test_size = 0.1, random_state = 0)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100)
forest = rfc.fit(X_train, Y_train)
forest.score(X_train, Y_train)


# In[ ]:


from sklearn import metrics
y_pred = forest.predict(X_test)
acc_test = metrics.accuracy_score(Y_test, y_pred)
acc_test


# In[ ]:


# place importances in a data frame
importances = pd.DataFrame({'feature':X_train.columns,
             'importance':np.round(forest.feature_importances_,3)})
# sort them by importance
importances = importances.sort_values('importance', ascending = False).set_index('feature')
importances
importances.plot.bar()


# These results are somewhat surprising, though maybe not really. By running the random forest, it appears as the satisfaction level is a huge motivating factor followed by the time spent with the company. Furthermore, there is a large gap in importance between the last evaluation of the employee and work accidents.
# 
# If you've come across this wondering why it isn't finished, I shall return! 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/HR_comma_sep.csv')
# Any results you write to the current directory are saved as output.

#assign numbers to categorical values
df.sales,df.salary=pd.Categorical(df.sales),pd.Categorical(df.salary)
df['sales_c'], df['salary_c'] = df.sales.cat.codes , df.salary.cat.codes
#print(df.sales.cat.categories,df.salary.cat.categories)
#create train and test dataset
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
train.shape,test.shape
#create db with separated for employees that left and stayed
a = df.loc[df['left'] == 1]
b = df.loc[df['left'] == 0]


# In[ ]:


#what about nans? ==> no NaNs
#print("number of NANS:\n", df.isnull().sum())

train.head()
train.drop(['sales','salary'],axis=1,inplace=True)
test.drop(['sales','salary'],axis=1,inplace=True)


#print(train.shape,test.shape)


# Data_exploration
# ----------------

# In[ ]:


#lets plot the categorical variables
lijst = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company', 'Work_accident','promotion_last_5years','sales_c','salary_c']
def create_histplot(lijst):
    a = df.loc[df['left'] == 1]
    b = df.loc[df['left'] == 0]
    FIG,AX,AX2=[],[],[]
    
    for i in range(1,len(lijst)+1):
        FIG.append(i); AX.append(i); AX2.append(i);
        
        FIG[i-1] = plt.figure()
        #make histogram subplot
        AX[i-1]=FIG[i-1].add_subplot(111)
        AX[i-1].hist(a[lijst[i-1]], alpha=0.95, color ='b'),AX[i-1].set_title(lijst[i-1]) 
        AX[i-1].hist(b[lijst[i-1]], alpha=0.55, color ='r')
        AX[i-1].hist(df[lijst[i-1]], alpha=0.35, color ='y')

        #plt.show()
        
        

create_histplot(lijst)   


# Satisfaction_level
# -three groups can be destinguished within the leavers:
# 	1. very unsatisfied
# 	2. intermediate satisfied
# 	3. satisfied
# -none of the leavers is perfectly satisfied in contrast to the stayers
# 
# Last_evaluation
# - leavers can be classified in two groups
# 	1. good evaluation
# 	2. mediocre (=bad) evaluation
# - a very bad evaluation is not equal to leaving, people get a second chance
# 
# Number_project
# - two groups can be seen in the leavers camp. one big group did only 2 projects
# - the other group did 4 or more, with maximums at 5 and 6
# 
# Average_Montly_hours
# - again two groups can be seen in the leavers
# 	1. that worked normal hours ~ 40 h workweek
# 	2. that worked a lot of hours ~ 54 - 80 h / week
# - more leavers than stayers worked extreme long working weeks
# 
# Time_spend_company
# - no leavers in the long within the long term >6 years workers
# - in the 5th year there are more leavers than stayers
# 
# Work_accident
# - a larger percentage of the eployees stayed after a work_accident.
# - in contrary to the expected result, work_accident doesn't appear to be positively correlated with leaving
# 
# 
# promotion_last_5years
# - only few people get a promotion (who?)
# - some of the people that got a promotion did leave
# 
# sales code ['IT'(0), 'RandD'(1), 'accounting'(2), 'hr'(3), 'management'(4), 'marketing'(5),
#        'product_mng'(6), 'sales'(7), 'support'(8), 'technical'(9)]
# 
# Salary code ['high'(0), 'low'(1), 'medium'(3)]

# In[ ]:


#sales - left; something strange happens with the xlabels
a = df.loc[df['left'] == 1]
b = df.loc[df['left'] == 0]

pl1 = plt.subplot2grid((1,2),(0,0))
a.sales.value_counts().plot(kind='bar', alpha =0.75)
b.sales.value_counts().plot(kind='bar', alpha =0.75)

pl2 = plt.subplot2grid((1,2),(0,1))
(a.sales.value_counts()/(df.sales.value_counts())).plot(kind='bar')


# In[ ]:


#salary - left
a = df.loc[df['left'] == 1]
b = df.loc[df['left'] == 0]


pl1 = plt.subplot2grid((1,2),(0,0))
a.salary.value_counts().plot(kind='bar', alpha =0.75)
b.salary.value_counts().plot(kind='bar', alpha =0.75)

pl2 = plt.subplot2grid((1,2),(0,1))
(a.salary.value_counts()/(df.salary.value_counts())).plot(kind='bar')


# In[ ]:


df.corr()


# In[ ]:


### three groups of leaving people can be identified: 
#I: good evaluation - low satisfaction
#II: medium evaluation - medium satisfaction
#III: high satisfaction - high evaluation

#however no garanty that they will leave

fig1=plt.figure()
ax1=fig1.add_subplot(211) #2*1 grid, first plot (211)
ax2=fig1.add_subplot(212) #2*1 grid, second plot (212)
ax1.scatter(df.loc[df['left'] == 1].satisfaction_level, df.loc[df['left'] == 1].last_evaluation)
plt.xlabel('Satisfaction_level') 
plt.ylabel('last_evaluation')

ax2.scatter(df.loc[df['left'] == 0].satisfaction_level, df.loc[df['left'] == 0].last_evaluation)
plt.xlabel('Satisfaction_level') 
plt.ylabel('last_evaluation')


# In[ ]:


#Two groups of leaving people can be seen:
#I: low evaluation - few hours
#II: high evaluation - many hours

fig2=plt.figure()
ax1=fig2.add_subplot(211) #2*1 grid, first plot (211)
ax2=fig2.add_subplot(212) #2*1 grid, second plot (212)
ax1.scatter(df.loc[df['left'] == 1].average_montly_hours, df.loc[df['left'] == 1].last_evaluation)
plt.xlabel('average_montly_hours') 
plt.ylabel('last_evaluation')

ax2.scatter(df.loc[df['left'] == 0].average_montly_hours, df.loc[df['left'] == 0].last_evaluation)
plt.xlabel('average_montly_hours') 
plt.ylabel('last_evaluation')


# In[ ]:


#similar groups to satisfaction level - last evaluation
fig3=plt.figure()
ax1=fig3.add_subplot(211) #2*1 grid, first plot (211)
ax2=fig3.add_subplot(212) #2*1 grid, second plot (212)
ax1.scatter(df.loc[df['left'] == 1].satisfaction_level, df.loc[df['left'] == 1].average_montly_hours)
plt.xlabel('satisfaction_level') 
plt.ylabel('average_montly_hours')

ax2.scatter(df.loc[df['left'] == 0].satisfaction_level, df.loc[df['left'] == 0].average_montly_hours)
plt.xlabel('satisfaction_level') 
plt.ylabel('average_montly_hours')


# 
# **handy commands cheat sheet:**
# 
#         df.head()
#         df.describe()
#          list(df)
#         #create training and test df
# 
#     

# In[ ]:


#lets make an svm

#first standardize data because we have varables in the 10^-1 range and 10^2 range
X_train=train.drop('left',axis=1) 
X_test=test.drop('left',axis=1)
Y_train=train.left
Y_test=test.left
#X_train = (X_train - X.mean()) /(X.max()-X.min())
def Normalizer (dataset):
    Norm = (dataset - dataset.mean()) /(dataset.max()-dataset.min())
    return Norm
X_train = Normalizer(X_train)
X_test = Normalizer(X_test)


# In[ ]:


#create the model wit sklearn
from sklearn import svm
clf = svm.SVC(kernel='rbf', C= 1, gamma = 20)
clf.fit(X_train,Y_train)


# In[ ]:


from sklearn import metrics
#predicted = clf.predict(X)
print("score on train dataset: \n",clf.score(X_train,Y_train))
print("score on test dataset: \n",clf.score(X_test,Y_test))


# In[ ]:


#lets do some cv to have an average of the performance of the svm
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, Y_train, cv=10)
print(scores)
    


# In[ ]:


#7% of the people that leave has been missed by our model
cnf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
print(cnf_matrix)
Y_train.value_counts()


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix
#7% of the people that leave has been missed by our model
cnf_matrix = confusion_matrix(Y_test, clf.predict(X_test))
print(cnf_matrix)
Y_test.value_counts()


# In[ ]:


#7% of the leavers missing is quite bad. Lets try to give more weight (overfit the data a little towards the leavers)
#this by dublicating all the leavers in the dataset
#However I don't think this will work for svm due to the nature of the model 
#as a hyperplane is searched in function of the two support vectors, defined by the data

#interesting would also be to know what leavers are misclassified, lets do that first


# In[ ]:


#misclassified leavers:
# no real pattern can be seen in the misclassified leavers, it would have been interesting if one of the
#three groups of above would have been visible
y_test = np.asarray(Y_test)
misclassified = np.where(y_test != clf.predict(X_test))
print(misclassified)
X_tes=test.drop('left',axis=1)
fig4=plt.figure()
ax1=fig4.add_subplot(111) #2*1 grid, first plot (211)
ax1.scatter(X_tes.iloc[misclassified].satisfaction_level, X_tes.iloc[misclassified].average_montly_hours)
plt.xlabel('satisfaction_level') 
plt.ylabel('average_montly_hours')


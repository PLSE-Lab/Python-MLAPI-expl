#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Project to identify Personal Loan Customers.
# 
# Bank is interested in converting its liability customers to asset customers by offering them personal loans.  
# A campaign run the previsous year showed a conversin rate of 9%.  This year they are interested in increasing the conversion
# rate by targeting high probaility customers.   This is a model to predict those customers that are highly likely to accept the 
# personal loan offer.

# ### Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

from scipy.stats import zscore


# ## Load Data

# In[ ]:


df = pd.read_csv('/kaggle/input/Bank_Personal_Loan_Modelling.csv')
df.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# In[ ]:


df.head(2)


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


# No columns have null data in the file
df.apply(lambda x : sum(x.isnull()))


# In[ ]:


# finding unique data
df.apply(lambda x: len(x.unique()))


# In[ ]:


df.describe().transpose()


# ## Observation on given data
# Min experience is showing negative values, which is incorrect, so we have to clean that column 

# ## Data distribution chart

# In[ ]:


plt = sns.pairplot(df[['Age','Experience','Income','ZIPCode','Family','CCAvg' ,'Education' , 'Mortgage','PersonalLoan','SecuritiesAccount','CDAccount','Online','CreditCard']] )
df.head(1)


# In[ ]:


# Observation on given data

# Age feature is normally distributed with majority of customers falling between 30 years and 60 years of age. 
# We can confirm this by looking at the describe statement above, which shows mean is almost equal to median

# Experience is normally distributed with more customer having experience starting from 8 years. 
# Here the mean is equal to median. There are negative values in the Experience. 
# This could be a data input error as in general it is not possible to measure negative years of experience. 
# We can delete these values.

# Income is positively skewed. Majority of the customers have income between 45K and 55K. 
# We can confirm this by saying the mean is greater than the median

# CCAvg is also a positively skewed variable and average spending is between 0K to 10K and 
# majority spends less than 2.5K

# Mortgage 70% of the individuals have a mortgage of less than 40K. However the max value is 635K

# The variables family and education are ordinal variables. The distribution of families is evenly distributes


# In[ ]:


plt = sns.boxplot(df[['Income']])#,'Experience','Income','ZIP Code','Family','CCAvg' ,'Education' , 'Mortgage','Personal Loan','Securities Account','CD Account','Online','CreditCard']] )


# In[ ]:


sns.distplot( df['Age'], color = 'r')


# ### Observation
# Most of the customers age fall in the age range of 30 to 60 yrs and their experience falls in the range of 5 to 35 years and most earn an income between 10K to 100K.

# ## Negative Experience cleaning

# In[ ]:


# Before "Negative Experience Cleaning"
# there are 52 records with negative value "Experience"
b4negExp = df.Experience < 0
b4negExp.value_counts()


# In[ ]:


dfposExp = df.loc[df['Experience'] >0]
mask = df.Experience < 0
column_name = 'Experience'
mylist = df.loc[mask]['ID'].tolist()


# In[ ]:


for id in mylist:
    age = df.loc[np.where(df['ID']==id)]["Age"].tolist()[0]
    education = df.loc[np.where(df['ID']==id)]["Education"].tolist()[0]
    df_filtered = dfposExp[(dfposExp.Age == age) & (dfposExp.Education == education)]
    exp = df_filtered['Experience'].median()
    df.loc[df.loc[np.where(df['ID']==id)].index, 'Experience'] = exp


# In[ ]:


# After "Negative Experience Cleaning"
# there are 0 records with negative value "Experience"
aftrnegExp = df.Experience < 0
aftrnegExp.value_counts()


# In[ ]:


df.describe().transpose()


# ## Influence of Income level on whether a customer takes a personal loan across the education levels. 

# In[ ]:


sns.boxplot(x="Education", y="Income", hue="PersonalLoan", data=df)


# ### Observation 
# The box plots show that those with education level 1 have higher incomes.  But customers who go for personal loans have the same income distribution regardless of the education level.

# In[ ]:


sns.boxplot(x="Education", y='Mortgage', hue="PersonalLoan", data=df)


# ### Observation
# From the above chart it seems that customer who do not have personal loan and customer who has personal loan have high mortgage

# In[ ]:


sns.countplot(x="ZIPCode", data=df[df.PersonalLoan==1], hue ="PersonalLoan",orient ='v')


# In[ ]:


zipcode_top5 = df[df.PersonalLoan==1]['ZIPCode'].value_counts().head(5)
zipcode_top5


# ### Observation
# Top 5 locations who appled personal loan before 

# In[ ]:


sns.countplot(x="Family", data=df,hue="PersonalLoan")


# ### Observations
# ### Does family size have any influence on whether a customer accepts a personal loan offer?

# In[ ]:


familysize_no = np.mean( df[df.PersonalLoan == 0]['Family'] )
familysize_no


# In[ ]:


familysize_yes = np.mean( df[df.PersonalLoan == 1]['Family'] )
familysize_yes


# In[ ]:


from scipy import stats

stats.ttest_ind(df[df.PersonalLoan == 1]['Family'], df[df.PersonalLoan == 1]['Family'])


# ### Observation 
# Family size seems to have no impact on decision to take a loan.

# In[ ]:


sns.countplot(x="SecuritiesAccount", data=df,hue="PersonalLoan")


# In[ ]:


# Observation : Majority of customers who does not have loan have securities account


# In[ ]:


sns.countplot(x="CDAccount", data=df,hue="PersonalLoan")


# In[ ]:


# Observation: Customers who does not have CD account , does not have loan as well. 
# This seems to be majority. But almost all customers who has CD account has loan as well


# In[ ]:


sns.countplot(x="CreditCard", data=df,hue="PersonalLoan")


# In[ ]:


sns.distplot( df[df.PersonalLoan == 0]['CCAvg'], color = 'r')
sns.distplot( df[df.PersonalLoan == 1]['CCAvg'], color = 'g')


# In[ ]:


print('Credit card spending of Non-Loan customers: ',df[df.PersonalLoan == 0]['CCAvg'].median()*1000)
print('Credit card spending of Loan customers    : ', df[df.PersonalLoan == 1]['CCAvg'].median()*1000)


# ### Observation
# Customers who have taken personal loan have higher credit card average than those who did nottake.  So high credit card average seems to be good predictor of whether or not a customer will take a personal loan.

# In[ ]:


sns.distplot( df[df.PersonalLoan == 0]['Income'], color = 'r')
sns.distplot( df[df.PersonalLoan == 1]['Income'], color = 'g')


# In[ ]:


sns.distplot( df[df.PersonalLoan == 0]['Education'], color = 'r')
sns.distplot( df[df.PersonalLoan == 1]['Education'], color = 'g')


# In[ ]:


from matplotlib import pyplot as plot
fig, ax = plot.subplots()
colors = {1:'red',2:'yellow',3:'green'}
ax.scatter(df['Experience'],df['Age'],c=df['Education'].apply(lambda x:colors[x]))
plot.xlabel('Experience')
plot.ylabel('Age')


# In[ ]:


# Observation The above plot shows experience and age have a positive correlation. 
# As experience increase age also increases. Also the colors show the education level. 
# There is gap in the mid forties of age and also more people in the under graduate level


# ## Correlation comparison with heat map

# In[ ]:


from matplotlib import pyplot as plt
plt.figure(figsize=(25, 25))
ax = sns.heatmap(df.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)
plt.title('Correlation')
plt.show()


# ### Observation 
# 1. Age and  Experoence is highly corelated
# 2. Income and CCAvg also corelated

# ### Splittin Data to Train And Test

# In[ ]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df.drop(['Experience' ,'ID' ,'CCAvg'], axis=1), test_size=0.3 , random_state=100)


# In[ ]:


train_set.describe().transpose()


# In[ ]:


test_set.describe().transpose()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
train_labels = train_set.pop("PersonalLoan")
test_labels = test_set.pop("PersonalLoan")


# ## DecisionTreeClassifier

# In[ ]:


dt_model = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 3)


# In[ ]:


dt_model.fit(train_set, train_labels)


# In[ ]:


dt_model.score(test_set , test_labels)


# In[ ]:


y_predict = dt_model.predict(test_set)
y_predict[:5]


# In[ ]:


test_set.head(5)


# ##  Naive Bayes

# In[ ]:


naive_model = GaussianNB()
naive_model.fit(train_set, train_labels)

prediction = naive_model.predict(test_set)
naive_model.score(test_set,test_labels)


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
cm = pd.DataFrame(confusion_matrix(test_labels, prediction).T, index=['No', 'Yes'], columns=['No', 'Yes'])
cm.index.name = 'Predicted'
cm.columns.name = 'True'
cm


# ## RandomForestClassifier

# In[ ]:


randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)
randomforest_model.fit(train_set, train_labels)


# In[ ]:


Importance = pd.DataFrame({'Importance':randomforest_model.feature_importances_*100}, index=train_set.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )


# In[ ]:


predicted_random=randomforest_model.predict(test_set)


# In[ ]:


randomforest_model.score(test_set,test_labels)


# ## KNeighborsClassifier

# In[ ]:


train_set_indep = df.drop(['Experience' ,'ID' ,'CCAvg'] , axis = 1).drop(labels= "PersonalLoan" , axis = 1)
train_set_indep_z = train_set_indep.apply(zscore)
train_set_dep = df["PersonalLoan"]
X = np.array(train_set_indep_z)
Y = np.array(train_set_dep)
X_Train = X[ :3500, :]
X_Test = X[3501: , :]
Y_Train = Y[:3500, ]
Y_Test = Y[3501:, ]


# In[ ]:


knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')
knn.fit(X_Train, Y_Train)    
predicted = knn.predict(X_Test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_Test, predicted)


# In[ ]:


print(acc)


# ### Model comparison

# In[ ]:


X=df.drop(['PersonalLoan','Experience','ID'],axis=1)
y=df.pop('PersonalLoan')


# In[ ]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=12345)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Conclusion
# The aim of the universal bank is to convert there liability customers into loan customers. They want to set up a new marketing campaign; hence, they need information about the connection between the variables given in the data. Four classification algorithms were used in this study. From the above graph , it seems like **Decision Tree** algorithm has the highest accuracy and we can choose that as our final model

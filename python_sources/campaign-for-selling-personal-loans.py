#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Bank is has a growing customer base. The bank wants to increase borrowers (asset customers) base to bring in more loan business and earn more through the interest on loans. So , bank wants to convert the liability based customers to personal loan customers. (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success.
# The department wants to build a model that will help them identify the potential customers who have higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.

# #### On the dataset 
# The file given below contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")

from scipy.stats import zscore
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection


# In[ ]:


data = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx','Data')
data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]


# In[ ]:


data.head()


# In[ ]:


data.columns


# #### Information on the features or attributes
# 
# The attributes can be divided accordingly :
# * The variable **ID** does not add any interesting information. There is no association between a person's customer ID  and loan, also it does not provide any general conclusion for future potential loan customers. We can neglect this information for our model prediction.
# 
# The binary category have five variables as below:
# 
# * Personal Loan - Did this customer accept the personal loan offered in the last campaign? ** This is our target variable**
# * Securities Account - Does the customer have a securities account with the bank?
# * CD Account - Does the customer have a certificate of deposit (CD) account with the bank?
# * Online - Does the customer use internet banking facilities?
# * Credit Card - Does the customer use a credit card issued by UniversalBank?
# 
# Interval variables are as below:
# 
# * Age - Age of the customer
# * Experience - Years of experience
# * Income - Annual income in dollars
# * CCAvg - Average credit card spending
# * Mortage - Value of House Mortgage
# 
# Ordinal Categorical Variables are:
# * Family - Family size of the customer
# * Education - education level of the customer
# 
# The nominal variable is :
# 
# * ID
# * Zip Code

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


# No columns have null data in the file
data.apply(lambda x : sum(x.isnull()))


# In[ ]:


# Eye balling the data
data.describe().transpose()


# In[ ]:


#finding unique data
data.apply(lambda x: len(x.unique()))


# In[ ]:


sns.pairplot(data.iloc[:,1:])


# * **Age** feature is normally distributed with majority of customers falling between 30 years and 60 years of age. We can confirm this by looking at the `describe` statement above, which shows **mean** is almost equal to **median**
# * **Experience** is normally distributed with more customer having experience starting from 8 years. Here the **mean** is equal to **median**. There are negative values in the **Experience**. This could be a data input error as in general it is not possible to measure negative years of experience. We can delete these values, because we have 3 or 4 records from the sample.
# * **Income** is positively skewed. Majority of the customers have income between 45K and 55K. We can confirm this by saying the **mean** is greater than the **median**
# * **CCAvg** is also a positively skewed variable and average spending is between 0K to 10K and majority spends less than 2.5K
# * **Mortgage**  70% of the individuals have a mortgage of less than 40K. However the max value is 635K
# * The variables family and education are ordinal variables. The distribution of families is evenly distributes

# In[ ]:


# there are 52 records with negative experience. Before proceeding any further we need to clean the same
data[data['Experience'] < 0]['Experience'].count()


# In[ ]:


#clean the negative variable
dfExp = data.loc[data['Experience'] >0]
negExp = data.Experience < 0
column_name = 'Experience'
mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience


# In[ ]:


# there are 52 records with negative experience
negExp.value_counts()


# The following code does the below steps:
# * For the record with the ID, get the value of `Age` column
# * For the record with the ID, get the value of `Education` column
# * Filter the records matching the above criteria from the data frame which has records with positive experience and take the median
# * Apply the median back to the location which had negative experience

# In[ ]:


for id in mylist:
    age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]
    education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]
    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]
    exp = df_filtered['Experience'].median()
    data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp


# In[ ]:


# checking if there are records with negative experience
data[data['Experience'] < 0]['Experience'].count()


# In[ ]:


data.describe().transpose()


# ### Influence of income and education on personal loan 

# In[ ]:


sns.boxplot(x='Education',y='Income',hue='PersonalLoan',data=data)


# **Observation** : It seems the customers whose education level is 1 is having more income. However customers who has taken the personal loan have the same income levels

# In[ ]:


sns.boxplot(x="Education", y='Mortgage', hue="PersonalLoan", data=data,color='yellow')


# **Inference** : From the above chart it seems that customer who do not have personal loan and customer who has personal loan have high mortgage

# In[ ]:


sns.countplot(x="SecuritiesAccount", data=data,hue="PersonalLoan")


# **Observation** : Majority of customers who does not have loan have securities account

# In[ ]:


sns.countplot(x='Family',data=data,hue='PersonalLoan',palette='Set1')


# **Observation**: Family size does not have any impact in personal loan. But it seems families with size of 3 are more likely to take loan. When considering future campaign this might be good association.

# In[ ]:


sns.countplot(x='CDAccount',data=data,hue='PersonalLoan')


# **Observation**: Customers who does not have CD account , does not have loan as well. This seems to be majority. But almost all customers who has CD account has loan as well

# In[ ]:


sns.distplot( data[data.PersonalLoan == 0]['CCAvg'], color = 'r')
sns.distplot( data[data.PersonalLoan == 1]['CCAvg'], color = 'g')


# In[ ]:


print('Credit card spending of Non-Loan customers: ',data[data.PersonalLoan == 0]['CCAvg'].median()*1000)
print('Credit card spending of Loan customers    : ', data[data.PersonalLoan == 1]['CCAvg'].median()*1000)


# **Observation**: The graph show persons who have personal loan have a higher credit card average. Average credit card spending with a median of 3800 dollar indicates a higher probability of personal loan.  Lower credit card spending with a median of 1400 dollars is less likely to take a loan. This could be useful information.

# In[ ]:


fig, ax = plot.subplots()
colors = {1:'red',2:'yellow',3:'green'}
ax.scatter(data['Experience'],data['Age'],c=data['Education'].apply(lambda x:colors[x]))
plot.xlabel('Experience')
plot.ylabel('Age')


# **Observation** The above plot show with experience and age have a positive correlation. As experience increase age also increases. Also the colors show the education level. There is gap in the mid forties of age and also more people in the under graduate level

# In[ ]:


# Correlation with heat map
import matplotlib.pyplot as plt
import seaborn as sns
corr = data.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
# create a mask so we only see the correlation values once
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask, 1)] = True
a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# **Observation**
# * Income and CCAvg is moderately correlated. 
# * Age and Experience is highly correlated

# In[ ]:


sns.boxplot(x=data.Family,y=data.Income,hue=data.PersonalLoan)
# Looking at the below plot, families with income less than 100K are less likely to take loan,than families with 
# high income


# ### Applying models
# Split data into train and test

# In[ ]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)


# In[ ]:


train_labels = train_set.pop('PersonalLoan')
test_labels = test_set.pop('PersonalLoan')


# ### Decision tree classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

dt_model=DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
dt_model.fit(train_set, train_labels)


# In[ ]:


dt_model.score(test_set , test_labels)


# In[ ]:


y_predict = dt_model.predict(test_set)
y_predict[:5]


# In[ ]:


test_set.head(5)


# ### Naive Bayes

# In[ ]:


naive_model = GaussianNB()
naive_model.fit(train_set, train_labels)

prediction = naive_model.predict(test_set)
naive_model.score(test_set,test_labels)


# ### Random Forest classifier

# In[ ]:


randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)
randomforest_model.fit(train_set, train_labels)


# In[ ]:


Importance = pd.DataFrame({'Importance':randomforest_model.feature_importances_*100}, index=train_set.columns)
Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )


# In[ ]:


predicted_random=randomforest_model.predict(test_set)
randomforest_model.score(test_set,test_labels)


# ### KNN ( K - Nearest Neighbour )

# In[ ]:


train_set_indep = data.drop(['Experience' ,'ID'] , axis = 1).drop(labels= "PersonalLoan" , axis = 1)
train_set_dep = data["PersonalLoan"]
X = np.array(train_set_indep)
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
print(acc)


# ### Model comparison

# In[ ]:


X=data.drop(['PersonalLoan','Experience','ID'],axis=1)
y=data.pop('PersonalLoan')


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
# The aim of the universal bank is to convert there liability customers into loan customers. They want to set up a new marketing campaign; hence, they need information about the connection between the variables given in the data. Four classification algorithms were used in this study. From the above graph , it seems like **Decision Tree** algorithm have the highest accuracy and we can choose that as our final model

# In[ ]:





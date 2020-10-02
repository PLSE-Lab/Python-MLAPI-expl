#!/usr/bin/env python
# coding: utf-8

# ### Import all required libraries

# In[ ]:


#Data Rendering
import numpy as np
import pandas as pd
import os
from pandas import Series, DataFrame

#visualization
import seaborn as sb
import matplotlib.pyplot  as plt
from pylab import rcParams

#Correlation libraries
import scipy 
from scipy.stats import spearmanr,chi2_contingency

#Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Setting for Matplotlib figure size

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=20,14
plt.style.use('seaborn-whitegrid')


# ### Load training and testing data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_address="/kaggle/input/titanic/train.csv"
test_address="/kaggle/input/titanic/test.csv"

train_DF=pd.read_csv(train_address)
test_DF=pd.read_csv(test_address)

combined_DF=[train_DF,test_DF]


# In[ ]:


train_DF.head()


# In[ ]:


test_DF.head()


# ### Some questions before starting the data analysis process
# 
# - Which age group people survived more?
#  > Age group between 20-40 survived 
# - Which Pclass survived more and how many male and female in that group?
#  > Pclass of 1 survived more compared to another classes
# - Which Embarked people survived?
#  > People of Embarked C i.e Cherbourg survived more.
# - How SibSp and Parch are correlated?
#  > Yes SibSp and Parch are correlated, this proved by using CHI-Square test

# ### Data Cleaning

# In[ ]:


train_DF.describe()


# In[ ]:


test_DF.describe()


# In[ ]:


survial_count=train_DF['Survived'].value_counts()
survive=survial_count[1]/(survial_count[0]+survial_count[1])*100
print(f'Total % people survied were {survive:0.2f}')


# We data of 891 people out of which 
# 
# Data is classified as shown below:
# - Categorical Columns: Survived, Sex, Embarked
# - Ordinal: Pclass
# - Numeric Columns: Age, Fare
# - MultiData: PassengerId, Name, Cabin
# - Discrete: SibSp, Parch
# 
# Now we have to find out relation between all features and Survived

# #### Which Sex people survived
# 
# - According to below observtions 74% Female survived whereas 18% Male Survived
# 
# 
# Total % of people survied were 38.4% out of which 0.742% were female.Which means female survival rate was more. This shows that survival rate and sex are correlated.

# In[ ]:


train_DF[['Survived','Sex']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# #### Which ticket class people survived more
# 
# - According to below analysis people of Pclass one survived 62% as compared to other class people.

# In[ ]:


train_DF[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_DF[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_DF[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_DF[['Embarked','Survived']].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Now we will try to find out relation between Age and Survived
# > So from below results we can say that people of age group 20-40 survived more.

# In[ ]:


g = sb.FacetGrid(train_DF, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


train_DF['AgeBand'] = pd.cut(train_DF['Age'], 5)
train_DF[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Remove columns that are not required
# - PassengerId,Name,Ticket,Fare,Cabin,AgeBand

# In[ ]:


train_DF=train_DF.drop(['PassengerId','Name','Ticket','Fare','Cabin','AgeBand'],axis=1)
test_DF=test_DF.drop(['Name','Ticket','Fare','Cabin'],axis=1)
train_DF.head()


# In[ ]:


train_DF.info()


# There are missing values in Age and Embarked Column.
# 
# So to fill missing values we will use:
# - Missing age values will be filled by median of all Ages
# - Missing Embarked values will be filled by median of Embarked feature.

# In[ ]:


frequent_embarked=train_DF['Embarked'].mode()[0]
frequent_embarked


# In[ ]:


#Fill NaN values of Embarked with 'S'
train_DF['Embarked']=train_DF["Embarked"].fillna(frequent_embarked)
#Fill NaN values of Age with 'median()'
train_DF['Age']=train_DF["Age"].fillna(train_DF['Age'].median())
##Fill NaN values of Age

test_DF['Embarked']=test_DF["Embarked"].fillna(frequent_embarked)
test_DF['Age']=test_DF["Age"].fillna(test_DF['Age'].median())


# We will now one-hotencode features Sex, Embarked and Age
# 
# - Sex=> Male=0,female=1
# - Embarked=> C=1,Q=2,S=3
# - Age is binned and values assigned are:
#   - 0-16=0
#   - 16-32=1
#   - 33-48=2
#   - 49-64=3
#   - above 64=4

# In[ ]:


title_mapping = {"male": 0, "female": 1, "C": 1, "Q": 2, "S": 3}
train_DF["Sex"]=train_DF["Sex"].map(title_mapping)
train_DF["Embarked"]=train_DF["Embarked"].map(title_mapping)
train_DF.loc[(train_DF['Age'] <= 16) , 'Age'] = 0
train_DF.loc[(train_DF['Age'] > 16) & (train_DF['Age'] <= 32), 'Age'] = 1
train_DF.loc[(train_DF['Age'] > 32) & (train_DF['Age'] <= 48), 'Age'] = 2
train_DF.loc[(train_DF['Age'] > 48) & (train_DF['Age'] <= 64), 'Age'] = 3
train_DF.loc[ train_DF['Age'] > 64, 'Age'] = 4 
test_DF["Sex"]=test_DF["Sex"].map(title_mapping)
test_DF["Embarked"]=test_DF["Embarked"].map(title_mapping)
test_DF.loc[(test_DF['Age'] <= 16) , 'Age'] = 0
test_DF.loc[(test_DF['Age'] > 16) & (test_DF['Age'] <= 32), 'Age'] = 1
test_DF.loc[(test_DF['Age'] > 32) & (test_DF['Age'] <= 48), 'Age'] = 2
test_DF.loc[(test_DF['Age'] > 48) & (test_DF['Age'] <= 64), 'Age'] = 3
test_DF.loc[ test_DF['Age'] > 64, 'Age'] = 4 
train_DF.head()


# In[ ]:


train_DF['Parch'].value_counts()


# ### Observations till now
# 
# - Data is now ready and in number format which is required by Machine Learning Model.
# - Now we would perform some tests to check correlation between features.
# - As our features contains ordinal values will be finding Spearman correlation

# ### Genrating Pairplot

# In[ ]:


sb.pairplot(train_DF)


# ### Spearmans Rank Correlation
# 
# #### Spearman correlation:
# - If p_value=1 => Strong positive correlation
# - If p_value=0 => No Correlation
# - If p_value=-1 => Strong negative correlation
# 
# #### CHI-SQUARE Test
# - To find out weather our features are independent or not we perform CHI-SQUARE Test
# 
# #### Observations
# - After performing spearman's rank correlation, we found out that all our feature Pclass,Sex,Age,SibSp,Parch, Embarked are not correlated.
# - After performing CHI-SQAURE Test we found that Pclass,Sex,Age,SibSp,Embarked are correlated while Parch is independent

# ### Spearman's Rank Correlation

# In[ ]:


pclass=test_DF["Pclass"]
sex=test_DF["Sex"]
age=test_DF["Age"]
sibsp=test_DF["SibSp"]
parch=test_DF["Parch"]
embarked=test_DF["Embarked"]

spearmanr_coefficient,p_value= spearmanr(pclass,sex)
print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')


# In[ ]:


spearmanr_coefficient,p_value= spearmanr(pclass,age)
print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')


# In[ ]:


spearmanr_coefficient,p_value= spearmanr(pclass,sibsp)
print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')


# In[ ]:


spearmanr_coefficient,p_value= spearmanr(pclass,parch)
print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')


# In[ ]:


spearmanr_coefficient,p_value= spearmanr(pclass,embarked)
print(f'Spearman Rank correlation coefficient {spearmanr_coefficient:0.3f}')


# ### CHI-SQUARE Test

# In[ ]:


table= pd.crosstab(pclass,sex)
chi2,p,dof,expected= chi2_contingency(table.values)
print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')


# In[ ]:


table= pd.crosstab(pclass,age)
chi2,p,dof,expected= chi2_contingency(table.values)
print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')


# In[ ]:


table= pd.crosstab(pclass,sibsp)
chi2,p,dof,expected= chi2_contingency(table.values)
print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')


# In[ ]:


table= pd.crosstab(pclass,parch)
chi2,p,dof,expected= chi2_contingency(table.values)
print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')


# In[ ]:


table= pd.crosstab(pclass,embarked)
chi2,p,dof,expected= chi2_contingency(table.values)
print(f'Chi-square statistic {chi2:0.3f} p_value{p:0.3f}')


# ### Separating Train and Test Feature

# In[ ]:


train_DF.head()


# ### Machine Learning Algorithms

# ### Splitting data in train and test set

# In[ ]:


X_train = train_DF.drop("Survived", axis=1)
Y_train = train_DF["Survived"]
X_test  = test_DF.drop(['PassengerId'],axis=1).copy()


# ### Creating Models

# ### LogisticRegression Model

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# ### Support Vector Classifier

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ### KNeighborsClassifier Model

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ### Decision Tree Model

# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ### Random Forest Model

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest',  
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# ### Conclusion
# 
# - After using various alogirthm, Random Forest and Decison Tree gave same accuracy score, we could be using Random Forest model with score of 86.31%

# In[ ]:


submission_df=pd.DataFrame({
    "PassengerId": test_DF["PassengerId"],
    "Survived": Y_pred_random
})
submission_df
# submission_df.to_csv('./submission.csv', index=False)


# ### References
# 
# - I have used below notebooks for reference, it helped me alot. 
#     - https://www.kaggle.com/startupsci/titanic-data-science-solutions
#     - https://www.kaggle.com/amarkumar2/titanic-predection-easy-solution
# 
# Please feel free to give your suggestions in comment section, and show your support by upvoting.
# 
# P.S: This is my first competition and I am open for feedback and connection.

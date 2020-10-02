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

import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from scipy import stats
import scipy.stats as stats
import pymc3 as pm
import arviz as az
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[ ]:


test_df.info()


# In[ ]:


new_train_df = train_df.drop(columns=['Survived'], axis=1)
total_df = pd.concat([new_train_df,test_df], sort=False, ignore_index=True)
total_df.isnull().sum()


# In[ ]:


total_df.head()


# In[ ]:


total_df['Title'] = total_df.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(total_df['Title'],total_df['Sex'])


# Since I catched that title is in relation to the age,; For example, Master is title given to boys and young men, and Miss is given to unmarried woman so that they tend to be younger;so I grouped the passengers by their title and see distributions repectively.
# 
# And you can check that there are meaningful differences between the titles.

# In[ ]:


list1 = ['Master','Mr']
list2 = ['Miss','Mrs','Ms','Mlle']

def AgeHist1(list1,list2,dataset):
    fig = plt.figure(figsize=(10,4))
    title1 = []
    title2 = []
    
    ax1 = fig.add_subplot(121)
    for i in np.arange(len(list1)):
        ax1 = sns.distplot(dataset['Age'].loc[dataset['Title']==list1[i]].dropna(),kde=False)
        ax1.set_ylabel('Counts')
        ax1.set_xlabel('Age')
        title1.append(list1[i])
    ax1.legend(labels=title1,loc='upper right',fontsize='small')
    ax2 = fig.add_subplot(122)
    for i in np.arange(len(list2)):
        ax2 = sns.distplot(dataset['Age'].loc[dataset['Title']==list2[i]].dropna(),kde=False)
        ax2.set_ylabel('Counts')
        ax2.set_xlabel('Age')
        title2.append(list2[i])
    ax2.legend(labels=title2,loc='upper right',fontsize='small')
        
AgeHist1(list1,list2,total_df)


# I grouped the titles by 6 categories. Particularly, ('Mlle','Miss','Lady') = Mr(Unmarried) and ('Mme','Ms','Mrs') = Mrs(Married) together. I know Ms is for woman who don't like to mention whether married or not, but grouped it with Mrs category anyways. :)

# In[ ]:


total_df['Title'] = total_df.Name.str.extract('([A-Za-z]+)\.',expand=False)
total_df['Title'] = total_df['Title'].replace(['Rev','Dr','Sir','Major','Countess'],'Special')
total_df['Title'] = total_df['Title'].replace(['Mlle','Miss','Lady'],'Miss')
total_df['Title'] = total_df['Title'].replace(['Mme','Ms','Mrs'],'Mrs')
total_df['Title'] = total_df['Title'].replace(['Col','Don','Dona','Jonkheer','Capt'],'The others')

pd.crosstab(total_df['Title'],total_df['Sex'])


# This is a indicator table where the number of True means there are missing values with that amount.

# In[ ]:


pd.crosstab(total_df['Title'],total_df['Age'].isna())


# In[ ]:


list = ['Master','Miss','Mr','Mrs']

def AgeHist2(list,dataset):
    fig = plt.figure(figsize=(10,6))
    
    for i in np.arange(len(list)):
        
        plt.subplot(math.ceil(len(list)/2),2,i+1)
        ax = sns.distplot(dataset['Age'].loc[dataset['Title']==list[i]].dropna(),kde=False)
        median = dataset['Age'].loc[dataset['Title']==list[i]].dropna().median()
        mean = dataset['Age'].loc[dataset['Title']==list[i]].dropna().mean()
        ax.axvline(mean, color='r', linestyle='--')
        ax.axvline(median, color='g', linestyle='--')
        ax.set_title('{}'.format(list[i]))
        plt.legend(labels=['mean','median'],loc='upper right',fontsize='small')
        plt.subplots_adjust(wspace=0.5, hspace=0.8)
        
AgeHist2(list,total_df)


# I tried to apply bayesian prediction by inferring posterior distribution from the dataset. First, I assume that the liklihood follows Gamma distribution because age is non-negative. It is always matter to choose proper prior because it affects posterior distribution. I have seen that there is no completely correct answer about this. :P Here I choose $\mu$ drawn from uniform distribution and $\sigma$ drawn from Halfnormal distribution with $10$ standard deviation.
# 
# $$\mu \sim Uniform(\min{Age}+\frac{\max{(Age|Title)}-\min{(Age|Title)}}{4}, \max{(Age|Title)}-\frac{\max{(Age|Title)}-\min{(Age|Title)}}{4})$$<br>
# $$\sigma \sim Halfnormal(\sigma = 10)$$<br>
# Likehood is $P((Age|Title)|\mu,\sigma)$
# You might know that Gamma distribution has two parameters known as $\alpha$ and $\beta$.<br>
# This can be calculated from $\mu$ and $\sigma$, and Gamma distribution from pymc3 allows to use these two parameters as the parameters for Gamma distribution, so I will use it.
# $$likelihood : Gamma(mu = \mu, sigma = \sigma, Observation = (Age|Title))$$
# 
# So the final goal is to find the optimal parameters $\mu$ and $\sigma$, and then we can draw a posterior gamma distribution from these parameters and sampling from it!
# 
# Since I catched that title is in relation to the age,; For example, Master is title given to boys and young men, and Miss is given to unmarried woman so that they are younger; I grouped the passengers by their title and inferred thier age distributions respectively.

# In[ ]:


def AgeBayesPredictor(title,dataset):
    
    missing = dataset['Age'].loc[dataset['Title']== title].isnull().sum()
    
    def AgeExtractor(title,dataset):
    
        Age = dataset['Age'].loc[dataset['Title']==title]
        Age = Age.dropna()
        
        return Age
    
    Age = AgeExtractor(title,dataset)
    

    with pm.Model() as model:
        
        upper = max(Age)-(1/4)*(max(Age)-min(Age))
        lower = min(Age)+(1/4)*(max(Age)-min(Age))
    
        #Set the prior
        mu = pm.Uniform('mu',upper = upper ,lower= lower)
        sigma = pm.HalfNormal('sigma',sd=10)
        
    
        #Liklihood
        observed = pm.Gamma('obs', mu=mu,sigma=sigma,observed=Age)
        
        
    with model:
        
        start = pm.find_MAP()
        
        #Trace
        trace = pm.sample(8000, start=start)
    
    #Sampling
    sampling = pm.sample_ppc(trace[1000:], model=model,samples=missing)
    sampling=[random.choice(sampling['obs'][i]) for i in np.arange(start=0, stop=missing)]
    return sampling

def imputeAge(title,dataset):
    
    for i in title:
        
        imputing_Age = AgeBayesPredictor(i,dataset)
        idx = dataset['Age'].loc[dataset['Title']==i].isnull()
        missing = dataset['Age'].loc[dataset['Title']==i][idx]
        
        for j in np.arange(len(missing)):
            missing.iloc[j] = imputing_Age[j]

        dataset.update(missing)
    return dataset

total_df = imputeAge(['Master','Miss','Mr','Mrs'],total_df)


# This is a distribution table after imputing age for each titles, and distribution does not change much as I expect.

# In[ ]:


list = ['Master','Miss','Mr','Mrs']
AgeHist2(list,total_df)


# After dropping passengers with missing cabins, I grouped the cabins by initial alphabats(For example, A12 = A, D22 = D and so on), and then I draw a conter plot to see the cabins assgined for each classes, and There is significant data imbalnce between Pclass 1 and Plcass2,3. I roughly assume that Pclass1 tend to be A, B, C and D Cabin and Pclass2 or 3 are tend to be E, F and G Cabin. As long as keeping in mind that I do not have enough cases from plcass2 and 3, this could be used as potential indicator to infer missing cabins for each classes.

# In[ ]:


train_Cabin_df = train_df.dropna(subset=["Cabin"])
#Cabin grouped by Initial Alphabats
def CategorizeCabin(data):
    
    for i in ['A','B','C','D','E','F','G','T']:
        Index = data["Cabin"].str.find(i)==0
        
        data["Cabin"][Index] = i
    
    return data

train_Cabin_df=CategorizeCabin(train_Cabin_df)
total_df=CategorizeCabin(total_df)
train_Cabin_df["Cabin"].unique()

#Update train data
train_df.update(train_Cabin_df)


# Passengers who use B, C, D and E are more likely to be survived, and these are mostly used by pclass1 passengers as we checked. This means that passengers(Pclass1) who use these cabins are tend to be survived than passengers who use F,G cabins which are used by Pclass3 mostly.
# There is also an exceptional cabin ,that is, E that all classes use. Survival rate is significantly high when they use E. There might be some information about this, but I will look into further.
# I think imputing missing Cabin is the one of important steps to achieve high score on this competion after my experince to try to fill out it. :)

# In[ ]:


fig = plt.plot()
sns.countplot(data=train_Cabin_df, x = "Pclass",hue ="Cabin")
plt.show()


# In[ ]:


fig = plt.figure
sns.countplot(data=train_Cabin_df, x ="Cabin", hue="Survived")
plt.show()
sns.barplot( x ="Cabin", y="Pclass",data=train_Cabin_df)
plt.show()
sns.countplot(data=train_Cabin_df, x ="Pclass", hue="Survived")
plt.show()
#sns.countplot(data=train_df, x ="Pclass", hue="Survived")
#plt.show()


# As you should see, There are postive correlation between Parch-Fare, Age-Fare, SibSp-Fare, and especially SibSp-Parch. This seems to be interpreted as passengers who have many Parch tend to get on a ship with their SibSp possibly. and negative correlation between Pclass-Fare, Pclass-Age and so on...
# Thses result seems natural to me. For example, Passengers from pclass1 pay more than thoes from Pclass3 and 2.
# Since SibSp and Parch are highly correlated each other, I can use one of them and also for Pclass and Fare.

# In[ ]:


corr = total_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

cmap = sns.diverging_palette(220,10,as_cmap = True)


sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# I will fill out trivial missing values below

# In[ ]:


#Fill missing NaN
total_df['Fare'] = total_df['Fare'].fillna(total_df['Fare'][total_df['Pclass']==3].mean())
total_df['Age'] = total_df['Age'].fillna(total_df['Age'][total_df['Title']=='Mr'].mean())
total_df['Embarked'] = total_df['Embarked'].fillna('S')


# In[ ]:


total_df.isnull().sum()


# In[ ]:


bins = np.linspace(min(total_df['Age'])-1,max(total_df['Age'])+1,num=6)
labels = ['Kid','Young Adult','Adult','Older Adult','Senior']
total_df['AgeGroup'] = pd.cut(total_df['Age'], bins=bins, labels=labels, right=False)


# In[ ]:


pd.crosstab(total_df['AgeGroup'],total_df['Title'])


# I dropped some features which are not necessary for train and prediction.

# In[ ]:


features_drop = ['PassengerId','Name', 'Ticket', 'Parch','Cabin','Age','Fare']
features = total_df.drop(features_drop, axis=1)
features.head()


# In[ ]:


#one-hot encoding
features = pd.get_dummies(features)
#separate train and label
train_label = train_df['Survived']
train_data = features.head(len(train_df))
test_data = features.tail(len(test_df))
train_data.head()


# In[ ]:


train_data, train_label = shuffle(train_data, train_label, random_state = 5)


# In[ ]:


def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction


# Random Forest shows the highest score, so I will choose it for my prediction finally.

# In[ ]:


# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
# Random Forest
rf_pred = train_and_test(RandomForestClassifier(n_estimators=50))
# Navie Bayes
nb_pred = train_and_test(GaussianNB())


# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission2 = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':rf_pred.astype(int)})

#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions2.csv'

submission2.to_csv(filename,index=False)

print('Saved file: ' + filename)


# To check if it is overfitted, I did 5-fold cross validation.
# We see that gaps between validation errors and train errors do not diverge. This means that the model is not overfitted inside of the train set at least. WE HAVE TO BE CAREFUL THAT IT IS NOT ALWAYS GOOD TO HAVE NEARLY PERFECT SCORE ON THE TRAIN SET BECAUSE IT MIGHT BE OVERFITTED TO THE TRAIN SET , SO IT HARMS TO PREDICT THE TEST SET.

# In[ ]:


cverror = []
trerror = []
for i in np.arange(5, 105, 5):
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(train_data, train_label)
    error1 = cross_val_score(clf,train_data,train_label, cv=5).mean()
    error2 = clf.score(train_data, train_label)
    cverror.append(1-error1)
    trerror.append(1-error2)
cverror = pd.DataFrame(cverror)
cverror.columns = ["cv-error"]
cverror["train-error"] = trerror
ax1=sns.lineplot(data=cverror)
ax1.set_title("5-fold cross validation")


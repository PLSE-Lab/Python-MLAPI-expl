#!/usr/bin/env python
# coding: utf-8

# # Predict if a customer will leave the company !
# 
# using the [chrurn dataset](https://www.kaggle.com/filemide/churns#churn_train.csv) investigate the features and there affect on the customer.
# 
# then build a logistic regression classifier to predict if a given customer will churn or not !
# 
# ![churn](/kaggle/input/churns/17328.329f1e71.630x354o.926138c0f9412x.jpeg)

# In[ ]:


# import first
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# change the style from the very beging
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### First let's import our dataset and take a look to it.

# In[ ]:


dataset = pd.read_csv("/kaggle/input/churns/churn_train.csv")
dataset.head()


# In[ ]:


dataset.dtypes


# In[ ]:


dataset['Customer ID']= dataset['Customer ID'].str.strip("ADF") 


# In[ ]:


dataset.describe()


# # Cleaning and preprocessing:

# ### Drop uneeded colmuns :

# In[ ]:


#dataset.drop(['Customer ID','network_age','Customer tenure in month'],inplace=True,axis=1)


# ## Handling the missing data

# In[ ]:


dataset.isna().sum()


# ## now we need to seperate the dependant and independent variables

# In[ ]:


dataset=pd.get_dummies(dataset,columns=['Most Loved Competitor network in in Month 1','Most Loved Competitor network in in Month 2','Network type subscription in Month 1','Network type subscription in Month 2'] )
features=dataset.drop(['Churn Status'],axis=1)
goal=dataset['Churn Status']
features


# In[ ]:


#standrize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features),columns=features.columns)
features


# # Logistic Regression

# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)


# ## let's build our model
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0,solver='lbfgs',multi_class='auto')
logistic.fit(train_set, goal_train)


# ## Well,, we need to evaluate that model !
# ## Let's use the confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, logistic.predict(test_set))
pd.DataFrame(cm)


# ## let's get the accuracy

# In[ ]:


print(logistic.score(train_set, goal_train))
print(logistic.score(test_set, goal_test))


# # K-Nearest-Neighbor classification 

# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)


# ## Time for the model !

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

accs_test = []
accs_train = []
ks = np.linspace(1, 30, 30)
for K in ks:
    classifier = KNeighborsClassifier(n_neighbors=int(K))
    classifier.fit(train_set, goal_train)
    cm = confusion_matrix(goal_test, classifier.predict(test_set))
    accs_train.append(classifier.score(train_set, goal_train))
    accs_test.append(classifier.score(test_set, goal_test))


# In[ ]:


plt.plot(ks, accs_train, label='train_acc')
plt.plot(ks, accs_test, label='test_acc')
plt.legend()
plt.title("accuracy versus K")


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(train_set, goal_train)


# ## take a look at the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, classifier.predict(test_set))
pd.DataFrame(cm)


# In[ ]:


print("model accuracy on train: {:.4f}".format(classifier.score(train_set, goal_train)))
print("model accuracy on test: {:.4f}".format(classifier.score(test_set, goal_test)))


# # Naive Bayes classification Tutorial

# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(train_set, goal_train)


# ## Let's see the confusion matrix first 

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, classifier.predict(test_set))
print(classifier.score(train_set, goal_train))

print(classifier.score(test_set, goal_test))

pd.DataFrame(cm)


# # `Bernoulli`
# 

# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB


# In[ ]:


clf = BernoulliNB()
clf.fit(train_set, goal_train)
clf.predict(test_set)


print(clf.score(train_set, goal_train))
clf.score(test_set, goal_test)


# # The best accuracy occured in Logistic Regression

# In[ ]:


test_dataset1= pd.read_csv("/kaggle/input/churns/churn_test.csv")
test_dataset1


# In[ ]:


test_dataset=pd.get_dummies(test_dataset1,columns=['Most Loved Competitor network in in Month 1','Most Loved Competitor network in in Month 2','Network type subscription in Month 1','Network type subscription in Month 2'] )


# In[ ]:


test_dataset['Customer ID']= test_dataset['Customer ID'].str.strip("ADF") 
#test_dataset['Network type subscription in Month 1_2']=0
#test_dataset['Network type subscription in Month 2_2']=0


# In[ ]:


#standrize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
test_dataset = pd.DataFrame(scaler.fit_transform(test_dataset),columns=test_dataset.columns)
test_dataset


# In[ ]:


test_dataset.columns


# In[ ]:


test_dataset


# In[ ]:


features=features.drop(['Network type subscription in Month 1_2','Network type subscription in Month 2_2'],axis=1)
# split into train and test
from sklearn.model_selection import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,test_size =0.2,random_state=0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state = 0,solver='lbfgs',multi_class='auto')
logistic.fit(train_set, goal_train)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, logistic.predict(test_set))
print(logistic.score(train_set, goal_train))
print(logistic.score(test_set, goal_test))
pd.DataFrame(cm)


# In[ ]:


ex=pd.read_csv('/kaggle/input/churns/sample submission.csv')
ex.dtypes


# In[ ]:


sub=pd.DataFrame({'Customer ID':test_dataset1.loc[:,'Customer ID'],'Churn Status':logistic.predict(test_dataset)})


# In[ ]:


sub['Customer ID'] = dataset['Customer ID'].str.replace("ADF","").astype(int)
sub.sort_values(by='Customer ID')
       


# In[ ]:


sub['Customer ID'] = sub['Customer ID'].astype(str)


# In[ ]:


for i in range(0,600,1):
    if int(sub.loc[i,'Customer ID']) < 100:
        sub.loc[i,'Customer ID'] = 'ADF00'+ sub.loc[i,'Customer ID']
    elif int(sub.loc[i,'Customer ID'])<1000:
        sub.loc[i,'Customer ID'] = 'ADF0'+ sub.loc[i,'Customer ID']
    else:
        sub.loc[i,'Customer ID'] = 'ADF'+ sub.loc[i,'Customer ID']
sub


# In[ ]:


sub.to_csv('submission.csv',index=False)


#!/usr/bin/env python
# coding: utf-8

# ### Importing all the libraries :

# In[1]:


# Data Analysis and Visulisation :
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Classification Libraries :
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ### Importing  data sets :

# In[2]:


train_df = pd.read_csv('../input/train.csv')


# In[3]:


test_df = pd.read_csv('../input/test.csv')


# In[4]:


# Sample :
train_df.head(4)


# In[5]:


test_df.head(4)


# ### Checking for NaN Values :

# In[6]:


print(train_df.isnull().sum())
sns.heatmap(train_df.isnull(),annot=False,yticklabels=False,cbar=False,cmap='summer')


# We can see there are two columns named 'Age' and 'Cabin' having Nan values.We have to handle it .

# In[7]:


train_df.drop(labels='Cabin',inplace=True,axis=1)
test_df.drop(labels='Cabin',inplace=True,axis=1)


# In[8]:


train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['Embarked'] = train_df['Embarked'].fillna('Q')
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())


# In[9]:


sns.heatmap(train_df.isnull(),annot=False,yticklabels=False,cbar=False,cmap='summer')


# Now there is no NaN values.

# ### checking NaN values for test data set :

# In[10]:


print(test_df.isnull().sum())


# column 'Fare' have NaN value.

# In[11]:


test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].mean()).astype(float)


# # Exploratory Data Analysis (EDA) :

# In[12]:


plt.figure(figsize=(10,6))
train_df['Age'].head(10).hist(bins=80,color = 'red',alpha=0.5)
plt.xlabel('Age',fontsize=12)
plt.ylabel('Number of passengers',fontsize = 12)
plt.show()


# In[13]:


plt.figure(figsize=(15,8))
plt.scatter(train_df.Survived,train_df.Age,color = 'green')
plt.ylabel('Age')
plt.title('Survival by Age(1 = Survived)')
plt.grid(b=True,which='major',axis = 'y')
plt.show()


# In[14]:


print(train_df.Pclass.value_counts())
plt.figure(figsize=(10,4.5))
plt.subplots_adjust(hspace = .2)
plt.subplot(1,2,1)
plt.title('Class Distribution',fontsize=12)
sns.countplot(data=train_df,x='Pclass',palette='winter')
plt.subplot(1,2,2)
plt.title('Class Distribution',fontsize=12)
train_df.Pclass.value_counts().plot(kind = 'pie',colormap='plasma')
plt.show()


# In[15]:


print(train_df.Survived.value_counts())
plt.figure(figsize=(10,4.5))
plt.subplots_adjust(hspace = .2)
plt.subplot(1,2,1)
plt.title('Survival breakdown ( 1 = Survived , 0 = Died)')
train_df.Survived.value_counts().plot(kind ='bar',colors=['pink','blue'])
plt.subplot(1,2,2)
plt.title('Survival breakdown ( 1 = Survived , 0 = Died)',fontsize=12)
train_df.Survived.value_counts().plot(kind = 'pie',colormap='spring')
plt.show()


# In[16]:


plt.figure(figsize=(15,4))
plt.subplots_adjust(hspace = .2)
plt.subplot(1,2,1)
sns.barplot(data=train_df,x='Survived',y='Age',hue='Sex',palette='cool')
plt.subplot(1,2,2)
sns.countplot(data=train_df,x='Pclass',hue='Sex',palette='cool')
plt.show()


# In[17]:


# corelaition matrics :
plt.figure(figsize=(10,5))
sns.heatmap(train_df.corr(),annot=True,cmap='winter')


# # Data Preprocessing :

# In[18]:


x = train_df.drop(['Survived', 'PassengerId','Name','Ticket'], axis=1).values


# In[19]:


x


# In[20]:


y = train_df["Survived"].values


# In[21]:


y


# In[22]:


# Encoding the Categorical values :
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
x[:,1]=encoder.fit_transform(x[:,1])
x[:,6]=encoder.fit_transform(x[:,6])


# In[23]:


# Spliting the Data :
from sklearn.cross_validation import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state = 0)


# In[24]:


# Converting features in same scale :
from sklearn.preprocessing import  StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.transform(xtest)


# # Modeling :

# ### Logistic Regression :

# In[25]:


from sklearn.linear_model import LogisticRegression
logistic_regressor = LogisticRegression(random_state=0)
logistic_regressor.fit(xtrain,ytrain)


# In[26]:


logistic_regressor.predict(xtest)


# In[27]:


logistic_accuracy = logistic_regressor.score(xtest,ytest)
logistic_accuracy


# ### K nearest neighbors :

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(xtrain, ytrain)


# In[29]:


knn.predict(xtest)


# In[30]:


knn_accuracy = knn.score(xtest,ytest)
knn_accuracy


# ### Support Vector Machines :

# In[31]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(xtrain,ytrain)


# In[32]:


svc.predict(xtest)


# In[33]:


svc_accuracy = svc.score(xtest,ytest)
svc_accuracy


# ### Perceptron :

# In[34]:


from sklearn.linear_model import Perceptron
perceptron_ob = Perceptron(random_state=5)
perceptron_ob.fit(xtrain, ytrain)


# In[35]:


perceptron_ob.predict(xtest)


# In[36]:


perceptron_accuracy = perceptron_ob.score(xtest,ytest)
perceptron_accuracy


# ### Naive Bayes :

# In[37]:


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain, ytrain)


# In[38]:


NB.predict(xtest)


# In[39]:


NB_accuracy = NB.score(xtest,ytest)
NB_accuracy


# ### Decision Tree Clasification :

# In[40]:


from sklearn.tree import DecisionTreeClassifier
dec_tree_classifier = DecisionTreeClassifier()
dec_tree_classifier.fit(xtrain, ytrain)


# In[41]:


dec_tree_classifier.predict(xtest)


# In[42]:


dec_tree_accuracy = dec_tree_classifier.score(xtest,ytest)
dec_tree_accuracy


# ### Random Forest Classification :

# In[43]:


from sklearn.ensemble import RandomForestClassifier
ran_forest_classifier = RandomForestClassifier()
ran_forest_classifier.fit(xtrain, ytrain)


# In[44]:


ran_forest_classifier.predict(xtest)


# In[45]:


ran_forest_accuracy = ran_forest_classifier.score(xtest,ytest)
ran_forest_accuracy


# ### Gradient Boost Classifier :

# In[46]:


from sklearn.ensemble import GradientBoostingClassifier
gr_boost_classifier = GradientBoostingClassifier()
gr_boost_classifier.fit(xtrain, ytrain)


# In[47]:


gr_boost_classifier.predict(xtest)


# In[48]:


gr_boost_accuracy = gr_boost_classifier.score(xtest,ytest)
gr_boost_accuracy


# ## Comparing the Performance of the Models :

# In[49]:


Models = ['Logistic Regression', 'KNN' , 'Support Vector Machines',
              'Perceptron', 'Naive Bayes','Decision Tree', 'Random Forest',
              'Gradient boosing']


# In[50]:


Accuracy = []

score = [logistic_accuracy, knn_accuracy, svc_accuracy, 
              perceptron_accuracy, NB_accuracy, dec_tree_accuracy, 
              ran_forest_accuracy, gr_boost_accuracy]

for i in score :
    Accuracy.append(round(i*100))
Accuracy


# In[51]:


Performance_of_Models = pd.DataFrame({'Model' : Models , 'Score' : Accuracy}).sort_values(by='Score', ascending=False)


# In[52]:


Performance_of_Models


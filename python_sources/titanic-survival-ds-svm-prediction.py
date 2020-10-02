#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival DS, Prediction by Support Vector Machine

# In[ ]:


import pandas as pd
import numpy as nm
import matplotlib as plt
import warnings


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


warnings.filterwarnings("ignore")


# In[ ]:


titan_ds = pd.read_csv("/kaggle/input/titanic/train.csv")
titan_ds.head()


# In[ ]:


titan_ds.shape


# In[ ]:


titan_ds.dtypes


# In[ ]:


titan_ds.columns


# In[ ]:


pd.value_counts(titan_ds['Survived'])


# In[ ]:


#replace the nans and nulls with median
median = round(titan_ds['Age'].median())
titan_ds['Age'] = titan_ds['Age'].fillna(value=median)
#Fill the cabin data in binary
titan_ds['Cabin'] = titan_ds["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#Fill the Sex data in binary
titan_ds['Sex'] = titan_ds['Sex'].apply(lambda x : 0 if x=='male' else 1)
titan_ds['FamilySize']=titan_ds['SibSp'] + titan_ds['Parch'] + 1
#create a new column for 'is_Alone' and fill if travelled alone. 
titan_ds['Is_Alone'] =titan_ds['FamilySize']==1
titan_ds['Is_Alone'] = titan_ds['Is_Alone'].apply(lambda x: 1 if x==1 else 0)


# In[ ]:


#Drop not required variables
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked']
titan_ds = titan_ds.drop(drop_elements, axis = 1)


# In[ ]:


pd.value_counts(titan_ds['Sex'])


# In[ ]:


print('missing values ',titan_ds.isna().sum())
print('missing values ',titan_ds.isnull().sum())


# In[ ]:


titan_ds.columns


# ### Why you should not travel Alone? Looks like there is 60% more chances of survival if you go with your family!

# #### Survival percentage of lone travellers:

# In[ ]:


len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1)])/ len(titan_ds[(titan_ds['Is_Alone'] == 1 )])


# #### Survival percentage of travel with partners:

# In[ ]:


len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) ])/ len(titan_ds[(titan_ds['Is_Alone'] != 1 )])


# ### Being Female, does they have more chance of survival? Looks like yes.

# In[ ]:


len(titan_ds[ (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[ (titan_ds['Sex'] == 1)])


# In[ ]:


len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[(titan_ds['Sex'] == 1) & (titan_ds['Is_Alone'] == 1 )])


# In[ ]:


len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[(titan_ds['Sex'] == 1) & (titan_ds['Is_Alone'] != 1 )])


# #### Worst victims are among lone Male passengers

# In[ ]:


##We could imagine women are 
len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 0)])/ len(titan_ds[(titan_ds['Sex'] == 0) & (titan_ds['Is_Alone'] != 1 )])


# In[ ]:


##We could imagine women are 
len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 0)])/ len(titan_ds[(titan_ds['Sex'] == 0) & (titan_ds['Is_Alone'] == 1 )])


# # Some Graphics now to proove the above numbers. 
# ### The Box plot reveals Being Male is not so lucky one in Titanic. 
# ### It does not matter even if you have a big family some times. 

# In[ ]:


plt.figure(figsize=(10,6))
ax=sns.boxplot('Sex','FamilySize',data=titan_ds,hue='Survived')
ax.set(xticklabels=['Male','Female']);


# In[ ]:


ax=sns.factorplot('Sex','FamilySize',data=titan_ds,hue='Survived')
ax.set(xticklabels=['Male','Female']);


# ### Does Age also played a factor of survival? Looks like to some extend. Survival rate is grim beyond 65s and beyond.

# In[ ]:


sns.stripplot(titan_ds['Survived'],titan_ds['Age']);


# In[ ]:


sns.swarmplot(titan_ds['Survived'],titan_ds['Age']);


# ### Let's verify the above throough correlation matrix. Does ticket price played any role? Looks like NO.

# In[ ]:


plt.figure(figsize=(10,5))
sns.heatmap(titan_ds.corr(), cmap='YlGnBu', annot=True)
plt.title("Titan survival heatmap")
plt.show();


# ### Time to Train our SVM model.  Linear Kernel with default c and default gamma

# In[ ]:


# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import numpy as np


#Get x and Y
X_train,y_train = np.array(titan_ds)[ :, 2:9], np.array(titan_ds.Survived)[:]

# Building a Support Vector Machine on train data
svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)


# #### Check the accuracy on the training set.

# In[ ]:


print(svc_model.score(X_train, y_train))


# Same cleanup like train data. I am lazy to not write a funcion :)

# In[ ]:


titan_test_ds= pd.read_csv("/kaggle/input/titanic/test.csv")
passenger_ids= titan_test_ds['PassengerId']
passenger_ids = passenger_ids.dropna()

#replace the nans and nulls with median
median = round(titan_test_ds['Age'].median())
titan_test_ds['Age'] = titan_test_ds['Age'].fillna(value=median)

titan_test_ds['Fare'] = titan_test_ds['Fare'].fillna(value=median)
#Fill the cabin data in binary
titan_test_ds['Cabin'] = titan_test_ds["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#Fill the Sex data in binary
titan_test_ds['Sex'] = titan_test_ds['Sex'].apply(lambda x : 0 if x=='male' else 1)
titan_test_ds['FamilySize']=titan_test_ds['SibSp'] + titan_test_ds['Parch'] + 1
#create a new column for 'is_Alone' and fill if travelled alone. 
titan_test_ds['Is_Alone'] =titan_test_ds['FamilySize']==1
titan_test_ds['Is_Alone'] = titan_test_ds['Is_Alone'].apply(lambda x: 1 if x==1 else 0)
#Drop not required variables
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked']
titan_test_ds = titan_test_ds.drop(drop_elements, axis = 1)


# ### Time to Test our SVM model and predict the target variable.

# In[ ]:


predictions = svc_model.predict(titan_test_ds.iloc[:,1:])
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('titanic_prediction.csv', index=False)


# # Verdict
# ### Our predicted score is 78%.
# ## Take Away:
# ### Using Support Vector Machine as a regression is good if you have multi colinearity dependent variables. As you can see there are features such age and sex. Where I felt like a conditional probability which increases chances of survival. So if you have features with multi colinearity and sparse data, Better to go with SVM. We've to improve by using better algos such as Random Forest or Ensamble techniques I think. Please do let's know if this was helpful.

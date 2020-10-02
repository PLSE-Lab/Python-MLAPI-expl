#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Please Don't forget to VoteUp if you like the kernel and the way of Code 
#Thank you :


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data1=pd.read_csv('../input/train.csv')


data1= pd.DataFrame(data=data1)
total_miss = data1.isnull().sum()
perc_miss = total_miss/data1.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


print(data1.shape)
data1.sort_values('PassengerId', inplace =True) 

data1.drop(['Cabin','Age'],axis=1,inplace=True)

print(data1.shape)
data1=data1.dropna()
print(data1.shape)




#data1.to_csv("clean_data1.csv",index=False, encoding='utf8')
data1.info()
data1.describe()


# In[ ]:





# In[ ]:



import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data2=pd.read_csv('../input/test.csv')


data2= pd.DataFrame(data=data2)
total_miss = data2.isnull().sum()
perc_miss = total_miss/data2.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


print(data2.shape)

data2.drop(['Cabin','Age'],axis=1,inplace=True)
mean_data2 = data2['Fare'].mean()
print(mean_data2)
data2['Fare'].fillna(mean_data2, inplace=True)


print(data2.shape)
data2=data2.dropna()
print(data2.shape)




#data1.to_csv("clean_data2.csv",index=False, encoding='utf8')
data1.info()
data1.describe()


# In[ ]:


category_column =['Name','Sex','Ticket','Embarked'] 
for x in category_column:
    print (x)
    print (data1[x].value_counts())
 


# In[ ]:


for col in category_column:
    b, c = np.unique(data1[col], return_inverse=True) 
    data1[col] = c

data1.head()


# In[ ]:


category_column =['Name','Sex','Ticket','Embarked'] 
for x in category_column:
    print (x)
    print (data2[x].value_counts())
 


# In[ ]:


for col in category_column:
    b, c = np.unique(data2[col], return_inverse=True) 
    data2[col] = c

data2.head()


# In[ ]:




import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data3=pd.read_csv('../input/gender_submission.csv')


data1= pd.DataFrame(data=data1)
total_miss = data3.isnull().sum()
perc_miss = total_miss/data3.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
train_pr=['PassengerId']
	

model = DecisionTreeRegressor()

#set prediction data to factors that will predict, and set target to SalePrice
train_data = data3[train_pr]
test_data = data2[train_pr]
target = data3['Survived']

#fitting model with prediction data and telling it my target
model.fit(train_data, target)

prediction=model.predict(test_data)
print(prediction)


# In[ ]:


prediction = prediction.reshape(len(prediction), 1)

dataTest = np.concatenate((data2, prediction), axis = 1)
print(dataTest)
data2['Survived'] = prediction
data2.sort_values('PassengerId', inplace =True) 

data2.head()


# In[ ]:



from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


#X=row_concat[['country of birth self','major occupation code','age','tax filer status']].values
X=data3[['PassengerId','Survived']].values


y= data2[['Survived']].values

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=902)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predn=clf.predict(X_test)
print('The accuracy of the model using decision tree is',metrics.accuracy_score(predn,y_test))


# In[ ]:


category_column =['Survived'] 
for x in category_column:
    print (x)
    print (data2[x].value_counts())


for col in category_column:
    b, c = np.unique(data2[col], return_inverse=True) 
    data2[col] = c

data2.head()     


# In[ ]:


data2.drop(['Pclass'	,'Name',	'Sex',	'SibSp',	'Parch',	'Ticket','Fare'	,'Embarked'],axis=1,inplace=True)


# In[ ]:



data2.to_csv("submission_output_2.csv",index=False, encoding='utf8')


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

hmap = data2.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

data2.hist(figsize = (10, 10))
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
plt.subplots(figsize=[10, 5])
data2['Survived'].value_counts().plot(kind='barh', title='Survived')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
survived=data2['Survived']

passengerid = data2['PassengerId']
plt.scatter( passengerid,survived, edgecolors='g')
plt.xlabel('Survived')
plt.ylabel('PASSENGER ID')
plt.title('PASSENGER ID With Total number of Survivor At the Titanic Ship Dataset')
plt.show()


# In[ ]:


import seaborn as sns
sns.boxplot(x='Survived',y='PassengerId',data=data2)


# In[ ]:


import seaborn as sns
sns.boxplot(x=data2['Survived'])


# In[ ]:


import seaborn as sns
sns.boxplot(x=data2['PassengerId'])


# In[ ]:


# Make a list of the column names to be plotted: cols
col = ['PassengerId','Survived']

# Generate the box plots
data2[col].plot(subplots=True,kind='box',figsize=[10,5])

# Display the plot
plt.show()


# In[ ]:


#Please Don't forget to VoteUp if you like the kernel and the way of Code 
#Thank you For Reviewing :-) 


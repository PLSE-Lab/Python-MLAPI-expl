#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Please Vote-up if you like the kernel and tell me in the comment About the kernel .Thank you for Reviewing


# In[ ]:




import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data1=pd.read_csv('../input/black-friday/train.csv')


data1= pd.DataFrame(data=data1)
total_miss = data1.isnull().sum()
perc_miss = total_miss/data1.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


print(data1.shape)
data1.sort_values('Purchase', inplace =True,ascending=False) 

data1.drop(['Product_Category_3'],axis=1,inplace=True)

print(data1.shape)




data1.to_csv("clean_train1.csv",index=False, encoding='utf8')
data1.info()
data1.describe()


# In[ ]:



import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data2=pd.read_csv('../input/black-friday/test.csv')


data2= pd.DataFrame(data=data2)
total_miss = data2.isnull().sum()
perc_miss = total_miss/data2.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


print(data2.shape)

data2.drop(['Product_Category_3'],axis=1,inplace=True)

print(data2.shape)




data2.to_csv("clean_test1.csv",index=False, encoding='utf8')
data2.info()
data2.describe()


# In[ ]:


mean_data1 = data1['Product_Category_2'].mean()
mean_data2 = data2['Product_Category_2'].mean()

data1['Product_Category_2'].fillna(mean_data1, inplace=True)
data2['Product_Category_2'].fillna(mean_data2, inplace=True)


# In[ ]:


print(mean_data1)
print(mean_data2)


# In[ ]:


data1.info()


# In[ ]:


category_column =['Product_ID','Gender','Age','City_Category','Stay_In_Current_City_Years'] 
for x in category_column:
    print (x)
    print (data1[x].value_counts())
 


for col in category_column:
    b, c = np.unique(data1[col], return_inverse=True) 
    data1[col] = c

print(data1.head())


# In[ ]:





# In[ ]:


data2.info()


# In[ ]:


category_column =['Product_ID','Gender','Age','City_Category','Stay_In_Current_City_Years'] 
for x in category_column:
    print (x)
    print (data2[x].value_counts())
 

for col in category_column:
    b, c = np.unique(data2[col], return_inverse=True) 
    data2[col] = c

data2.head()


# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeRegressor
train_pr=['Gender','Age','Occupation','City_Category']


model = DecisionTreeRegressor()

#set prediction data to factors that will predict, and set target to SalePrice
train_data = data1[train_pr]
test_data = data2[train_pr]
target = data1.Purchase

#fitting model with prediction data and telling it my target
model.fit(train_data, target)

prediction=model.predict(test_data)
print(prediction)


# In[ ]:


prediction = prediction.reshape(len(prediction), 1)

dataTest = np.concatenate((data2, prediction), axis = 1)
print(dataTest)
data2['purchase prediction'] = prediction
data2.sort_values('purchase prediction', inplace =True,ascending=False) 

data2.head()


# In[ ]:


data2.to_csv("predicted_test1.csv",index=False, encoding='utf8')
data2.info()
data2.describe()


# In[ ]:


print("Original ")
print(data1.shape)

print("\nAfter removing")
data1 = data1.iloc[:233599]

print(data1.shape)


# In[ ]:


#due to continuous values i have to make it integer so the values will easy to predict the accuracy
category_column =['purchase prediction'] 
for x in category_column:
    print (x)
    print (data2[x].value_counts())


for col in category_column:
    b, c = np.unique(data2[col], return_inverse=True) 
    data2[col] = c

data2.head()     


# In[ ]:





# In[ ]:



from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics


#X=row_concat[['country of birth self','major occupation code','age','tax filer status']].values
X=data1[['Purchase']].values


y= data2[['purchase prediction']].values

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=902)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predn=clf.predict(X_test)
print('The accuracy of the model using decision tree is',metrics.accuracy_score(predn,y_test))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

hmap = data2.corr()
plt.subplots(figsize=(22,35))
sns.heatmap(hmap, vmax=2,annot=True,cmap="BrBG", square=True);


# In[ ]:


data1.hist(figsize = (20, 20))
plt.show()


# In[ ]:


data2.hist(figsize = (20, 20))
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Age',y='purchase prediction',data=data2,hue='Gender' ,palette='coolwarm')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Age')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Occupation',y='purchase prediction',data=data2,hue='Gender' ,palette='winter')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Occupation')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='City_Category',y='purchase prediction',data=data2,hue='Gender' ,palette='spring')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with City_Category ')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Stay_In_Current_City_Years',y='purchase prediction',data=data2,hue='Gender' ,palette='coolwarm')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Stay_In_Current_City_Years')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Marital_Status',y='purchase prediction',data=data2,hue='Gender' ,palette='summer')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Marital_Status')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Product_Category_1',y='purchase prediction',data=data2,hue='Gender' ,palette='autumn')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Product_Category_1')

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(12, 12))
g = sns.barplot(x='Product_Category_2',y='purchase prediction',data=data2,hue='Gender' ,palette='coolwarm')
g = g.set_ylabel("Purchase Amount of Prediction")
plt.title('Total Purchase Amount of Prediction with Product_Category_2')

plt.show()


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Age',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Gender',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Occupation',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='City_Category',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Stay_In_Current_City_Years',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Marital_Status',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Product_Category_1',y='purchase prediction',data=data2)


# In[ ]:


import seaborn as sns
#plt.subplots(figsize=[22, 30])

sns.boxplot(x='Product_Category_2',y='purchase prediction',data=data2)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
age=data2['Age']

pp = data2['purchase prediction']
plt.scatter(age,pp, edgecolors='r')
plt.xlabel('age')
plt.ylabel('purchase prediction')
plt.title('age with purchase prediction')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
oc=data2['Occupation']

pp = data2['purchase prediction']
plt.scatter(oc,pp, edgecolors='r')
plt.xlabel('Occupation')
plt.ylabel('purchase prediction')
plt.title('Occupation with purchase purchase Prediction')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
oc=data2['Gender']

pp = data2['purchase prediction']
plt.scatter(oc,pp, edgecolors='r')
plt.xlabel('Gender')
plt.ylabel('purchase prediction')
plt.title('Gender with purchase purchase Prediction')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
oc=data2['Marital_Status']

pp = data2['purchase prediction']
plt.scatter(oc,pp, edgecolors='r')
plt.xlabel('Marital_Status')
plt.ylabel('purchase prediction')
plt.title('Marital_Status with purchase purchase Prediction')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
oc=data2['Product_Category_1']

pp = data2['purchase prediction']
plt.scatter(oc,pp, edgecolors='r')
plt.xlabel('Product_Category_1')
plt.ylabel('purchase prediction')
plt.title('Product_Category_1 with purchase purchase Prediction')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
oc=data2['Product_Category_2']

pp = data2['purchase prediction']
plt.scatter(oc,pp, edgecolors='r')
plt.xlabel('Product_Category_2')
plt.ylabel('purchase prediction')
plt.title('Product_Category_2 with purchase purchase Prediction')
plt.show()


# In[ ]:


#Please Vote-up if you like the code and give me the comment .Thank you Reviewing 


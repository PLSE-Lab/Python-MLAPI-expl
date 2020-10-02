#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Data = pd.read_csv("../input/german_credit_data.csv")
print (Data.columns)
Data.head(10)


# In[ ]:


print("Purpose : ",Data.Purpose.unique())
print("Sex : ",Data.Sex.unique())
print("Housing : ",Data.Housing.unique())
print("Saving accounts : ",Data['Saving accounts'].unique())
print("Checking account : ",Data['Checking account'].unique())


# In[ ]:



Data['Saving accounts'] = Data['Saving accounts'].map({"little":0,"moderate":1,"quite rich":2 ,"rich":3 });
Data['Saving accounts'] = Data['Saving accounts'].fillna(Data['Saving accounts'].dropna().mean())

Data['Checking account'] = Data['Checking account'].map({"little":0,"moderate":1,"rich":2 });
Data['Checking account'] = Data['Checking account'].fillna(Data['Checking account'].dropna().mean())

Data['Sex'] = Data['Sex'].map({"male":0,"female":1}).astype(float);

Data['Housing'] = Data['Housing'].map({"own":0,"free":1,"rent":2}).astype(float);

Data['Purpose'] = Data['Purpose'].map({'radio/TV':0, 'education':1, 'furniture/equipment':2, 'car':3, 'business':4,
       'domestic appliances':5, 'repairs':6, 'vacation/others':7}).astype(float);

Data.head(10)


# In[ ]:


plt.scatter(Data['Credit amount'],Data["Age"])
plt.figure()


# In[ ]:


sns.pairplot(Data)


# In[ ]:


plt.scatter(Data['Credit amount'],Data["Duration"])
plt.figure()


# In[ ]:


plt.scatter(Data['Saving accounts'],Data["Duration"])
plt.figure()


# In[ ]:


fig = Data["Purpose"].hist(bins=8)
fig.text(-1, 150, 'Frequency', ha='center')
fig.text(0, -30, 'Radio', ha='center')
fig.text(1, -50, 'education', ha='center')
fig.text(2, -30, 'furniture', ha='center')
fig.text(3, -50, 'car', ha='center')
fig.text(4, -30, 'business', ha='center')
fig.text(5, -50, 'appliances', ha='center')
fig.text(6, -30, 'repairs', ha='center')
fig.text(7, -50, 'vacation', ha='center')


# In[ ]:


limitedCredit = Data[(Data["Credit amount"]<=5000)==True];
imitedCredit = Data[(Data["Credit amount"]>2000)==True];
fig = limitedCredit["Purpose"].hist(bins=8)
fig.text(-1, 150, 'Frequency', ha='center')
fig.text(0, -30, 'Radio', ha='center')
fig.text(1, -50, 'education', ha='center')
fig.text(2, -30, 'furniture', ha='center')
fig.text(3, -50, 'car', ha='center')
fig.text(4, -30, 'business', ha='center')
fig.text(5, -50, 'appliances', ha='center')
fig.text(6, -30, 'repairs', ha='center')
fig.text(7, -50, 'vacation', ha='center')


# In[ ]:


fig = Data.Age.hist(bins=60)
fig.text(40, -10, 'Age', ha='center')
fig.text(0, 40, 'Frequency', ha='center')


# In[ ]:


fig = Data["Job"].hist()
fig.text(-0.5, 400, 'Frequency', ha='center')
fig.text(0, -100, 'UnSkilled', ha='center')
fig.text(1, -100, 'UnSkilled Resident', ha='center')
fig.text(2, -100, 'Skilled', ha='center')
fig.text(3, -100, 'Highly Skilled', ha='center')


# ## Result:
#     
#     1. People from Age 23 to 32 are the target customer and the amount can be in range 2000 to 5000 <currency>.
#     2. Offers for car loan and radio loan can pick up more customers or lenders.
#     3. Short term credit with credit range 2000 t0 5000 yield maximum customer and profits.

# In[ ]:


from sklearn.cluster import KMeans;
from sklearn.decomposition import PCA; 
from sklearn.preprocessing import normalize;
y = KMeans().fit_predict(Data)
X_norm = normalize(Data);
y_PCA = PCA(n_components=2).fit_transform(X_norm,2);
y_PCA.shape


# In[ ]:


plt.scatter(Data['Credit amount'],Data['Age'],c=y)
plt.figure()
plt.scatter(y_PCA[:,0],y_PCA[:,1],c=y)


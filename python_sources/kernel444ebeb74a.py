#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
df


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns.tolist()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


#for rating of resturant url,adress,phone,listed_in(type)not needed so to be dropped
df=df.drop(['url', 'address','phone','listed_in(city)'], axis=1)
df


# In[ ]:


df.name.value_counts().head()


# In[ ]:


df.name.value_counts().tail()


# In[ ]:


plt.figure(figsize = (13,8))
r = df.name.value_counts()[:50].plot(kind = 'bar',color='green',fontsize=15)
r.legend(['Restaurants'])
plt.xlabel("Name of Restaurant")
plt.ylabel("Count of Restaurants")
plt.title("Name vs count of Restaurant",fontsize =15, weight = 'bold',color='red')


# In[ ]:


#no of  accepting online ordesrs
df.online_order.value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.online_order.value_counts().plot(kind = 'bar',color='yellow',fontsize=15)
r.legend(['orders'])
plt.xlabel("online orders")
plt.ylabel("Count ")
plt.title("No of online orders",fontsize =15, weight = 'bold',color='red')


# # online orders are more

# In[ ]:


df['book_table'].value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.book_table.value_counts().plot(kind = 'bar',color='red',fontsize=15)
r.legend(['book table'])
plt.xlabel("book_table")
plt.ylabel("no of resturants ")
plt.title("book table facility",fontsize =15, weight = 'bold',color='blue')


# In[ ]:


#location
df['location'].value_counts()[:15]


# In[ ]:


plt.figure(figsize = (13,8))
s= df.location.value_counts()[:20].plot(kind = 'bar',color='pink',fontsize=25)
r.legend(['location'])
plt.xlabel("location")
plt.ylabel("count ")
plt.title("location vs count",fontsize =15, weight = 'bold',color='blue')


# In[ ]:


df['rest_type'].value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.rest_type.value_counts()[:20].plot(kind = 'bar',color='lightskyblue',fontsize=25)
r.legend(['rest type'])
plt.xlabel("rest_type")
plt.ylabel("count ")
plt.title("rest type vs count",fontsize =15, weight = 'bold',color='blue')


# In[ ]:


#rename approx cost column
df.rename(columns={'approx_cost(for two people)': 'approx_cost'}, inplace=True)


# In[ ]:


df['approx_cost'].value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.approx_cost.value_counts()[:20].plot(kind = 'bar',color='orange',fontsize=25)
r.legend(['approx_cost'])
plt.xlabel("approx_cost")
plt.ylabel("count ")
plt.title("approx cost vs count",fontsize =15, weight = 'bold',color='blue')


# In[ ]:


df=df[df.dish_liked.isna()==False]


# In[ ]:


df.isna().sum()


# In[ ]:


df['dish_liked'].value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.dish_liked.value_counts()[:20].plot(kind = 'bar',color='lightgreen',fontsize=25)
r.legend(['dish liked'])
plt.xlabel("dish_liked")
plt.ylabel("count ")
plt.title("approx cost vs count",fontsize =15, weight = 'bold',color='blue')


# biriyani most liked

# In[ ]:


df['rates'].value_counts()


# In[ ]:


df=df[df.rates.isna()==False]


# In[ ]:


df['rates'].value_counts()


# In[ ]:


plt.figure(figsize = (13,8))
s= df.rates.value_counts()[:20].plot(kind = 'bar',color='lightgreen',fontsize=25)
r.legend(['rates'])
plt.xlabel("rates")
plt.ylabel("count ")
plt.title("rates vs count",fontsize =15, weight = 'bold',color='blue')


# #average rating is 3.9
# 

# In[ ]:


df['cuisines'].value_counts()


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x=df['rates'], hue = df['online_order'])
plt.ylabel("Restaurants that Accept/Not Accepting online orders")
plt.title("rate vs online order",weight = 'bold')


# In[ ]:


df['location'].nunique()


# # 87 locations from where resturants accessed through zomato
# 

# In[ ]:


#creating dummies for online order,table booked as it contains categorical yes and no
df['online_order']= pd.get_dummies(df.online_order, drop_first=True)
df['book_table']= pd.get_dummies(df.book_table, drop_first=True)
df


# In[ ]:


df.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)


# In[ ]:


df['rest_type'] = df['rest_type'].str.replace(',' , '') 
df['rest_type'] = df['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
df['rest_type'].value_counts().head()


# In[ ]:


df['cuisines'] = df['cuisines'].str.replace(',' , '') 
df['cuisines'] = df['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
df['cuisines'].value_counts().head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
T = LabelEncoder()                 
df['location'] = T.fit_transform(df['location'])
df['rest_type'] = T.fit_transform(df['rest_type'])
df['cuisines'] = T.fit_transform(df['cuisines'])
#df['dish_liked'] = T.fit_transform(df['dish_liked'].


# In[ ]:


df["approx_cost"] = df["approx_cost"].astype(str).str.replace(',' , '') 


# In[ ]:


df["approx_cost"] =df["approx_cost"].astype('float')


# In[ ]:


df.head()


# In[ ]:


x = df.drop(['rates','name','approx_cost'],axis = 1)
x


# In[ ]:


y=df['rates']
y


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)
x_train


# In[ ]:


x_test


# In[ ]:


x_test.fillna(x_train.mean(), inplace=True)


# In[ ]:


col_mask=df.isnull().any(axis=0) 


# In[ ]:


row_mask=df.isnull().any(axis=1)


# In[ ]:


df.loc[row_mask,col_mask]


# In[ ]:


np.isnan(x.values.any())


# In[ ]:


df=df[df.approx_cost.isna()==False]


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred_lr = lr.predict(x_test)


# In[ ]:


lr.score(x_test, y_test)*100


# In[ ]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred_rfr = rfr.predict(x_test)


# In[ ]:


rfr.score(x_test,y_test)*100


# In[ ]:


##SVM
from sklearn import metrics
from sklearn.svm import SVC
s= SVC()
s.fit(x_train,y_train)
y_pred_s = s.predict(x_test)  


# In[ ]:


s.score(x_test,y_test)*100


# In[ ]:





# In[ ]:





# In[ ]:





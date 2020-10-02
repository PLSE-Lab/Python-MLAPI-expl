#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/udemy-courses/clean_dataset.csv")


# # Data

# In[ ]:


data.head()


# In[ ]:


data.describe()


# # unique course title

# In[ ]:


len(data['course_title'].value_counts())


# In[ ]:


data.shape


# In[ ]:


data_paid= data[data['is_paid']==True]


# # 3368 paid courses

# In[ ]:


data_paid.shape


# In[ ]:


data_paid.head()


# In[ ]:


data_free=data[data['is_paid']==False]


# # 310 free courses

# In[ ]:


data_free.shape


# In[ ]:


data_free.head()


# # order according to suscribers

# In[ ]:


data_free.sort_values(by='num_subscribers',ascending=False)


# In[ ]:


data_paid.sort_values(by='num_subscribers',ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.scatterplot(x="price", y="num_subscribers",hue="num_subscribers",ax=ax ,data=data_paid).set(title = 'price vs subscribers(paid)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# # course with highest number of subscribers

# In[ ]:


data_paid[data_paid['num_subscribers']==max(data_paid['num_subscribers'])]


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.scatterplot(x="price", y="num_lectures",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'price vs number of lectures(paid)',xlabel= "price")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_paid['subject'].value_counts()


# In[ ]:


data_paid[data_paid['price']=='200']['subject'].value_counts()


# In[ ]:


data_free['subject'].value_counts()


# # Number of paid courses in each subject

# In[ ]:


sns.countplot(x='subject', data=data_paid)


# # Number of free courses in each subject

# In[ ]:


sns.countplot(x='subject', data=data_free)


# In[ ]:


import re

data[data['course_title'].str.contains(r'Data')== True]


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.scatterplot(x="price", y="engagement",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'price vs engagement(paid)')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# # Courses with highest engagement

# In[ ]:


data_paid[data_paid['engagement']==1.0]


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.set_palette("Blues_d")
sns.scatterplot(x="num_lectures", y="engagement",hue="num_lectures",ax=ax ,data=data_paid).set(title = 'engagement vs number of lectures(paid)')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_paid[data_paid['num_lectures']==max(data_paid['num_lectures'])]


# In[ ]:


sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.scatterplot(x="num_subscribers", y="num_reviews",hue="num_reviews",ax=ax ,data=data_paid).set(title = 'price vs number of lectures(paid)')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_paid_10=data_paid.sort_values(by='num_subscribers',ascending=False)[0:10].sort_values("num_subscribers", ascending=False).reset_index(drop=True).reset_index()[['course_id','course_title','num_subscribers','num_reviews','price']]


# In[ ]:


data_paid_10


# In[ ]:



sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.barplot(x="course_title", y="num_subscribers",ax=ax ,data=data_paid_10).set(title = 'price vs number of lectures(paid)')
plt.xticks(rotation=90)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_free_10=data_free.sort_values(by='num_subscribers',ascending=False)[0:10].sort_values("num_subscribers", ascending=False).reset_index(drop=True).reset_index()[['course_id','course_title','num_subscribers','num_reviews','price']]


# In[ ]:


data_free_10


# In[ ]:



sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
fig.set
sns.barplot(x="course_title", y="num_subscribers",ax=ax ,data=data_free_10).set(title = 'price vs number of lectures(paid)')
plt.xticks(rotation=90)


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_paid['subject'].value_counts()


# In[ ]:


data_paid_business = data_paid[data_paid['subject']=='Business Finance']


# In[ ]:


data_paid_business['price']=data_paid_business['price'].apply(lambda x:int(x))
type(data_paid_business['price'][0])


# # Distribution of price across different subjects

# In[ ]:


sns.distplot(data_paid_business['price'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# In[ ]:


data_paid_development = data_paid[data_paid['subject']=='Web Development']


# In[ ]:


data_paid_development['price']=data_paid_development['price'].apply(lambda x:int(x))


# In[ ]:


sns.distplot(data_paid_development['price'])


# In[ ]:


data_paid_musical = data_paid[data_paid['subject']=='Musical Instruments']


# In[ ]:


data_paid_musical['price']=data_paid_musical['price'].apply(lambda x:int(x))


# In[ ]:


sns.distplot(data_paid_musical['price'])


# # Highest paid Musical Courses

# In[ ]:


data_paid_musical[data_paid_musical['price']==200]


# In[ ]:


data1=data


# In[ ]:


datat=data1.drop(['course_id','course_title','url','num_reviews','published_timestamp','engagement','content_multiplier'],axis=1)


# In[ ]:


data.isnull().sum()


# In[ ]:


datat.head()


# In[ ]:


data['level'].value_counts()


# In[ ]:


sns.countplot(x='level',data=data)


# In[ ]:


datat.corr()


# In[ ]:


y = datat['num_subscribers']
x = datat.drop(['num_subscribers'],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


x['is_paid'] = x['is_paid'].apply(lambda x:str(x))
x['price'] = x['price'].apply(lambda x: 0 if x=='Free' else x)
x['price'] = x['price'].apply(lambda x:int(x))


# In[ ]:


#x['price'] = scaler.fit_transform(x.price.values.reshape(-1, 1))


# In[ ]:


#x['num_lectures'] = scaler.fit_transform(x.num_lectures.values.reshape(-1, 1))


# In[ ]:


#x['content_duration'] = scaler.fit_transform(x.content_duration.values.reshape(-1, 1))


# In[ ]:


#x['content_time_value'] = scaler.fit_transform(x.content_time_value.values.reshape(-1, 1))


# In[ ]:


x = pd.get_dummies(x)


# In[ ]:


x.columns


# In[ ]:


from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split


# In[ ]:


regressor =  RandomForestRegressor(n_estimators = 100, random_state = 0) 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[ ]:


regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


feat_importances = pd.Series(regressor.feature_importances_, index=x.columns)
feat_importances.plot(kind='barh')


# In[ ]:


from sklearn.metrics import mean_absolute_error as mse
mse_sub = mse(y_pred,y_test)


# In[ ]:


mse_sub


# In[ ]:



from xgboost import XGBRegressor


# In[ ]:


regressor = XGBRegressor()
regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


mse_sub = mse(y_pred,y_test)


# In[ ]:


mse_sub


# In[ ]:


feature_important = regressor.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
data.plot(kind='barh')


# In[ ]:





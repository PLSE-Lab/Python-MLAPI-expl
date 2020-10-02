#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sm


# In[ ]:


df = pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')


# In[ ]:


df.head(5)


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:





# In[ ]:


df.columns


# In[ ]:


df.dtypes.value_counts()


# In[ ]:


sns.countplot(data=df, x = 'rate', label='Count')

df['rate'].value_counts()


# In[ ]:


import seaborn as sns
sns.countplot(data=df, x = 'votes', label='Count')

df['votes'].value_counts()


# In[ ]:


plt.figure(figsize = (12,6))
ax = df.name.value_counts()[:20].plot(kind = 'bar')
ax.legend(['* Restaurants'])
plt.xlabel("Name of Restaurant")
plt.ylabel("Count of Restaurants")
plt.title("Name vs Number of Restaurant",fontsize =20, weight = 'bold')


# In[ ]:


ax= sns.countplot(df['online_order'])
plt.title('Number of Restaurants accepting online orders', weight='bold')
plt.xlabel('online orders')


# In[ ]:


sns.set_context("paper", font_scale = 1, rc = {"font.size": 20,"axes.titlesize": 25,"axes.labelsize": 20})   
b = sns.countplot(data = df, x = 'listed_in(city)', hue = 'book_table')
plt.title('Number of restaurants in each city in which you can book a table or not')
b.set_xticklabels(b.get_xticklabels(),rotation = 90)
plt.show()


# In[ ]:


df['location'].value_counts()


# In[ ]:


plt.figure(figsize=(12,6)) 
df['location'].value_counts()[:10].plot(kind = 'pie')
plt.title('Location', weight = 'bold')


# In[ ]:


df['approx_cost(for two people)'].value_counts()[:20]


# In[ ]:


plt.figure(figsize = (12,6))
df['location'].value_counts()[:10].plot(kind = 'bar', color = 'g')
plt.title("Location vs Count", weight = 'bold')


# In[ ]:


plt.figure(figsize = (12,6))
sns.countplot(x=df['rate'], hue = df['online_order'])
plt.ylabel("Restaurants that Accept/Not Accepting online orders")
plt.title("rates vs oline order",weight = 'bold')


# In[ ]:


df = df.drop(columns=['url','address','phone','location','reviews_list','menu_item','name'],axis=1)


# In[ ]:


df.describe()


# In[ ]:


df[df.duplicated(keep=False)]


# In[ ]:


df.drop_duplicates(keep=False,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.replace(0,np.nan,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df_new = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})


# In[ ]:


df_new.head(20)


# In[ ]:


df_new.columns


# In[ ]:


df_new.info()


# In[ ]:


df_new.isna().sum()


# In[ ]:


mean_value=df_new['votes'].mean()
df_new['votes']=df_new['votes'].fillna(mean_value)


# In[ ]:


df_new.isna().sum()


# In[ ]:


df_new.isnull().sum()


# In[ ]:


df['dish_liked'].unique()


# In[ ]:


df['rest_type'].unique()


# In[ ]:


df['rate'].unique()


# In[ ]:


df_new.head()


# In[ ]:


df_new = df_new.loc[df_new.rate !='NEW']
df_new = df_new.loc[df_new.rate !='-'].reset_index(drop=True)


# In[ ]:


#df_new['rates'] = df_new['rates'].apply(lambda x: str(x).split('/')[0])
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
df_new.rate = df_new.rate.apply(remove_slash).str.strip().astype('float')


# In[ ]:


df_new['rate'].head()


# In[ ]:


df_new.isna().sum()


# In[ ]:


df_new.info()


# In[ ]:


df_new.dropna(how='any',inplace=True)


# In[ ]:


df_new.head()


# In[ ]:





# In[ ]:


df_new.shape


# In[ ]:


#Some Transformations
df_new['cost'] = df_new['cost'].astype(str)
df_new['cost'] = df_new['cost'].apply(lambda x: x.replace(',','.'))
df_new['cost'] = df_new['cost'].astype(float)
df_new.info()


# In[ ]:


df_new.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lr=LabelEncoder()

for i in df_new.select_dtypes("object").columns:
    df_new[i]=lr.fit_transform(df_new[i])


# In[ ]:


df_new


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


# In[ ]:





# In[ ]:


X = df_new.drop(['rate'],1)
y = df_new['rate']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=42)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[ ]:


df_new.head()


# In[ ]:


from sklearn.linear_model import LinearRegression 
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:





# In[ ]:




from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)


# In[ ]:





rfr.score(X_test,y_test)*100




# In[ ]:


from sklearn.linear_model import Lasso
ls = Lasso()
ls.fit(X_train,y_train)
y_pred_ls = ls.predict(X_test)


# In[ ]:


ls.score(X_test,y_test)*100


# In[ ]:


from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train,y_train)
y_pred_rdg = rdg.predict(X_test)


# In[ ]:


rdg.score(X_test,y_test)*100


# In[ ]:


import pandas as pd
ridgepred = pd.DataFrame({ "actual": y_test, "pred": y_pred_rfr })
ridgepred


# In[ ]:





# In[ ]:





# In[ ]:





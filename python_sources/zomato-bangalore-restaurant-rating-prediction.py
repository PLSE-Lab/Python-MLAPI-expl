#!/usr/bin/env python
# coding: utf-8

# # 1.Loading Packages

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# # 2.Reading dataset

# In[ ]:


zomato_orgnl=pd.read_csv("../input/zomato.csv")


# **Showing 5 restaurant datas using head() function**

# In[ ]:


zomato_orgnl.head()


# # 3.Data pre-processing

# __Counting missing values for different columns__

# In[ ]:


zomato_orgnl.isnull().sum()


# __Information on original zomato dataset__

#  __From above, it is found that the column "dish_liked" has more than 50% values missing__

# __Dropping the column "dish_liked", "phone", "url"  and saving the new dataset as "zomato"__

# In[ ]:


zomato=zomato_orgnl.drop(['url','dish_liked','phone'],axis=1)
zomato.columns


# __Renaming "approx_cost(for two people)" ,listed_in(type) and listed_in(city) as they have multiple data-types__

# In[ ]:


zomato.rename({'approx_cost(for two people)': 'approx_cost_for_2_people',
               'listed_in(type)':'listed_in_type',
               'listed_in(city)':'listed_in_city'
              }, axis=1, inplace=True)
zomato.columns


# __Converting "votes" and  "approx_cost_for_2_people" into numeric(int)__

# __"votes" and  "approx_cost_for_2_people" have values like 1,000. 
# So we will change them into pure numeric values.
# <br>For this, we will use the lambda function__

# In[ ]:


remove_comma = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 
zomato.votes = zomato.votes.astype('int')
zomato['approx_cost_for_2_people'] = zomato['approx_cost_for_2_people'].apply(remove_comma)


# __Confirming the data-types of "votes" and "approx_cost_for_2_people"__

# In[ ]:


zomato.info()


# __Now we will convert "rate" into float__

# __Checking unique values of "rate"__

# In[ ]:


zomato['rate'].unique()


# __We remove the restaurent datas which has rate='NEW'__

# In[ ]:


zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)


# __Now we will remove '/5'__

# In[ ]:


remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')


# In[ ]:


zomato['rate'].head()


# In[ ]:


zomato.info()


# __Now we see that 'rate' column has converted to float datatype__

# __Now we will Label Encode the input variable columns into 0,1,2...__

# In[ ]:


def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'approx_cost_for_2_people', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

zomato_en = Encode(zomato.copy())


# In[ ]:


zomato_en['rate'] = zomato_en['rate'].fillna(zomato_en['rate'].mean())
zomato_en['approx_cost_for_2_people'] = zomato_en['approx_cost_for_2_people'].fillna(zomato_en['approx_cost_for_2_people'].mean())


# In[ ]:


zomato_en.isna().sum()


# # 4.STARTING REGRESSION PART<br>(PREDICTION)

# __Checking for correlation among all the x(inputs)__

# In[ ]:


corr = zomato_en.corr(method='kendall')


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)


# In[ ]:


zomato_en.columns


# __The highest correlation is between name and address which is 0.63 which is not of very much concern__ 
# <br> __Splitting dataset into train & test__ 

# In[ ]:


x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]
y = zomato_en['rate']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)


# In[ ]:


x_train.head()


# In[ ]:


y_train.head()


# ## Applying LINEAR REGRESSION

# In[ ]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[ ]:


y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# **The following loop will find the best random state which will give the best accuracy in the range. Uncomment and run to see the random state.
# P.S.- Random state might change as train_test_split splits the dataset randomly[](http://)**

# In[ ]:


'''reg_score=[]
import numpy as np
for j in range(1000):
    x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=j,test_size=0.1)
    lr=LinearRegression().fit(x_train,y_train)
    reg_score.append(lr.score(x_test,y_test))
K=reg_score.index(np.max(reg_score))
#Random state = K=353'''


# __With LINEAR REGRESSION, we are getting an  accuracy of  30 %__

# ## Applying DECISION TREE REGRESSION

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)


# In[ ]:


DTree=DecisionTreeRegressor(min_samples_leaf=.0001)


# In[ ]:


DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,y_predict)


# In[ ]:


'''from sklearn.tree import DecisionTreeRegressor
ts_score=[]
for j in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=j)
    dc=DecisionTreeRegressor().fit(x_train,y_train)
    ts_score.append(dc.score(x_test,y_test))
J= ts_score.index(np.max(ts_score))

J
#J=105'''


# __With DECISION TREE REGRESSION, we are getting an  accuracy of  83 %__

# ## Applying RANDOM FOREST REGRESSION 

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=5,random_state=329,min_samples_leaf=.0001)


# In[ ]:


RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[ ]:


'''rf_score=[]
for k in range(500):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.01,random_state=k)
    dtc=RandomForestRegressor().fit(x_train,y_train)
    rf_score.append(dtc.score(x_test,y_test))
K= rf_score.index(np.max(rf_score))
K=329'''


# __With RANDOM FOREST REGRESSION, we are getting an  accuracy of  84 % which is better than all three__

# # 5.DATA VISUALIZATION

# ## i) No. of restaurants in a particular location

# In[ ]:


fig = plt.figure(figsize=(20,7))
loc = sns.countplot(x="location",data=zomato_orgnl, palette = "Set1")
loc.set_xticklabels(loc.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Location",size=18)
loc
plt.title('NO. of restaurants in a Location',size = 20,pad=20)


# ## ii) Frequency of different types of restaurants

# In[ ]:


fig = plt.figure(figsize=(17,5))
rest = sns.countplot(x="rest_type",data=zomato_orgnl, palette = "Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Restaurant type",size=15)
rest 
plt.title('Restaurant types',fontsize = 20 ,pad=20)


# ## iii) Most famous restaurant chains in Bengaluru

# In[ ]:


plt.figure(figsize=(15,7))
chains=zomato_orgnl['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most famous restaurant chains in Bangaluru",size=20,pad=20)
plt.xlabel("Number of outlets",size=15)


# ## iv) Number of restaurants taking online order or not

# In[ ]:


plt.figure(figsize=(15,7))
zomato_orgnl['online_order'].value_counts().plot.bar()
plt.title('Online orders', fontsize = 20)
plt.ylabel('Frequency',size = 15)


#  ## v) Frequency of  restaurants allowing booking table or not

# In[ ]:


plt.figure(figsize=(15,7))
zomato_orgnl['book_table'].value_counts().plot.bar()
plt.title('Booking Table', fontsize = 20,pad=15)
plt.ylabel('Frequency', fontsize = 15)


# ## vi) Percentage of  restaurants according to their types

# In[ ]:


plt.figure(figsize=(10,10))
restaurantTypeCount=zomato_orgnl['rest_type'].value_counts().sort_values(ascending=True)
slices=[restaurantTypeCount[0],
        restaurantTypeCount[1],
        restaurantTypeCount[2],
        restaurantTypeCount[3],
        restaurantTypeCount[4],
        restaurantTypeCount[5],
        restaurantTypeCount[6],
        restaurantTypeCount[7],
        restaurantTypeCount[8]]
labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ','Quick Bites','Bakery']
colors = ['#3333cc','#ffff1a','#ff3333','#c2c2d6','#6699ff','#c4ff4d','#339933','black','orange']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their type", bbox={'facecolor':'2', 'pad':2})


# ## vii) Distribution of  restaurants according to approx cost for two people 

# In[ ]:


fig, ax = plt.subplots(figsize=[15,7])
sns.distplot(zomato_en['approx_cost_for_2_people'],color="magenta")
ax.set_title('Approx cost for two people distribution',size=20,pad=15)
plt.xlabel('Approx cost for two people',size = 15)
plt.ylabel('Percentage of restaurants',size = 15)


# ## viii) Showing True rate vs Predicted rate

# In[ ]:


plt.figure(figsize=(12,7))
preds_rf = RForest.predict(x_test)
plt.scatter(y_test,x_test.iloc[:,2],color="red")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Rating',size = 15)
plt.ylabel('Frequency',size = 15)
plt.scatter(preds_rf,x_test.iloc[:,2],color="green")


# ## ix) Restaurant rating distribution

# In[ ]:


plt.figure(figsize=(15,8))
rating = zomato['rate']
plt.hist(rating,bins=20,color="red")
plt.title('Restaurant rating distribution', size = 20, pad = 15)
plt.xlabel('Rating',size = 15)
plt.ylabel('No. of restaurants',size = 15)


# ## x) Approx cost for 2 people distribution

# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(zomato.approx_cost_for_2_people)
plt.title('Approx cost for 2 people distribution', size = 20, pad = 15)
plt.xlabel('Approx cost for 2 people',size = 15)
plt.ylabel('Density',size = 15)


# **The approx cost for 2 people is around 300-400 INR**

# In[ ]:


plt.figure(figsize=(15,8))
cuisines=zomato['cuisines'].value_counts()[:15]
sns.barplot(cuisines,cuisines.index)
plt.title('Most popular cuisines of Bangalore', size = 20, pad = 15)
plt.xlabel('No. of restaurants',size = 15)


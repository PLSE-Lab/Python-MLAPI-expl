#!/usr/bin/env python
# coding: utf-8

# # Complete Data Visualization and EDA for Zomato Restaurants in Bangalore City

# Import the libraries

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression #Logistic Regression is a Machine Learning classification algorithm
from sklearn.linear_model import LinearRegression #Linear Regression is a Machine Learning classification algorithm
from sklearn.model_selection import train_test_split #Splitting of Dataset
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# import the dataset

# In[ ]:


dataset = pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")
dataset.head()


# # Data Preprocessing

# In[ ]:


dataset.shape


# In[ ]:


dataset.dtypes


# Let's count missing values for different column

# In[ ]:


dataset.isnull().sum()


# From above, it is found that the column "dish_liked" has more than 50% values missing
# 
# Dropping the column "dish_liked" "url","phone"  and saving the new dataset as "zomato"

# In[ ]:


zomato=dataset.drop(['url','dish_liked','phone'],axis=1)
zomato.columns


# Renaming "approx_cost(for two people)" ,listed_in(type) and listed_in(city) as they have multiple data-types

# In[ ]:


zomato.rename({'approx_cost(for two people)': 'approx_cost_for_2_people',
               'listed_in(type)':'listed_in_type',
               'listed_in(city)':'listed_in_city'
              }, axis=1, inplace=True)
zomato.columns


# Converting "votes" and "approx_cost_for_2_people" into numeric(int)
# 
# "votes" and "approx_cost_for_2_people" have values like 1,000. So we will change them into pure numeric values.
# For this, we will use the lambda function

# In[ ]:


remove_comma = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 
zomato.votes = zomato.votes.astype('int')
zomato['approx_cost_for_2_people'] = zomato['approx_cost_for_2_people'].apply(remove_comma)


# In[ ]:


zomato.info()


# We remove the restaurent datas which has rate='NEW'

# In[ ]:


zomato = zomato.loc[zomato.rate !='NEW']
zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)


# Now we will remove '/5' from rate

# In[ ]:


remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')


# In[ ]:


zomato['rate']


# Now we will Label Encode the input variable columns into 0,1,2...

# In[ ]:


def Encode(zomato):
    for column in zomato.columns[~zomato.columns.isin(['rate', 'approx_cost_for_2_people', 'votes'])]:
        zomato[column] = zomato[column].factorize()[0]
    return zomato

zomato_en = Encode(zomato.copy())


# now filling the null vaslues in the columns "rate" and "approx_cost_for_2_people" with their mean.

# In[ ]:


zomato_en['rate'] = zomato_en['rate'].fillna(zomato_en['rate'].mean())
zomato_en['approx_cost_for_2_people'] = zomato_en['approx_cost_for_2_people'].fillna(
                                         zomato_en['approx_cost_for_2_people'].mean())


# In[ ]:


zomato_en.isna().sum()


# # Let's start a Regression part

# 1. Heat Map
# 2. Linear Regression
# 3. Decision Tree Regression
# 4. Random Forrest Regression

# # Checking for correlation among all the column(inputs)

# In[ ]:


plt.figure(figsize=(14,7))
sns.heatmap(zomato_en.corr(method='kendall'), annot=True )


# The highest correlation is between name and address which is 0.63.

# # Splitting dataset into train & test

# In[ ]:


x = zomato_en.iloc[:,[2,4,5,6,7,8]]
y = zomato_en['rate']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)


# In[ ]:


x_train.head()


# # Applying LINEAR REGRESSION Model

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# # Applying DECITION TREE REGRESSION Model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
DecTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DecTree.fit(x_train,y_train)
y_predict=DecTree.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# # Applying RANDOM FOREST REGRESSION Model

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=5,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict1=RForest.predict(x_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_predict1)


# **hence we found far better result from decision tree and random forest regression rather than linear regression model.**

# # Data vizualisation and Expolatory Data Analysis for the given data

# # 1. Visualizing Linear relationship 

# here we see that, lmplot() function in seaborn is used to visualize a linear relationship as determined through regression.

# In[ ]:


sns.set(color_codes=True)


# # Linear Relationship between rates and approx_cost_for_2_people shown below

# In[ ]:


sns.lmplot(x="rate",y="approx_cost_for_2_people", data=zomato);


# In[ ]:


sns.lmplot(x="rate",y="approx_cost_for_2_people",hue="online_order", data=zomato);


# # Linear Relationship between rate and votes shown below:

# In[ ]:


sns.lmplot(x="rate",y="votes", data=zomato);


# In[ ]:


sns.lmplot(x="rate",y="votes", hue="book_table",data=zomato);


# # 2. No. of restaurants in a particular location

# In[ ]:


fig = plt.figure(figsize=(20,7))
loc = sns.countplot(x="location",data=dataset, palette = "Set1")
loc.set_xticklabels(loc.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Location",size=18)
loc
plt.title('NO. of restaurants in a Location',size = 20,pad=20)


# # 3. Number of restaurants taking online order or not

# here, we are using plotly to show the result.

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
x=dataset['online_order'].value_counts()
colors = ['#FEBFB3', '#E1396C']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="Accepting vs not accepting online orders",width=500,height=500)
fig=go.Figure(data=[trace],layout=layout)
py.iplot(fig, filename='pie_chart_subplots')


# # 4. Frequency of restaurants allowing booking table or not

# here, we are using plotly to show the result.

# In[ ]:


x=dataset['book_table'].value_counts()
colors = ['Blue', '#E1396C']

trace=go.Pie(labels=x.index,values=x,textinfo="value",
            marker=dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout=go.Layout(title="booking vs not booking tables online",width=500,height=500)
fig=go.Figure(data=[trace],layout=layout)
py.iplot(fig, filename='pie_chart_subplots')


# # 5. Most famous restaurant chains in Bengaluru

# In[ ]:


plt.figure(figsize=(15,7))
chains=zomato['name'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='Set1')
plt.title("Most famous restaurant chains in Bangaluru",size=20,pad=20)
plt.xlabel("Number of outlets",size=15)


# # 6. Percentage of restaurants according to their types

# In[ ]:


plt.figure(figsize=(10,10))
restaurantTypeCount=zomato['rest_type'].value_counts().sort_values(ascending=True)
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
colors = ['#3333cc','#ffff1a','#ff3333','#c2c2d6','#6699ff','#c4ff4d','#339933','pink','orange']
plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)
fig = plt.gcf()
plt.title("Percentage of Restaurants according to their type", bbox={'facecolor':'2', 'pad':2})


# # 7. Frequency of different types of restaurants

# In[ ]:


fig = plt.figure(figsize=(17,5))
rest = sns.countplot(x="rest_type",data=zomato, palette = "Set1")
rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")
plt.ylabel("Frequency",size=15)
plt.xlabel("Restaurant type",size=15)
rest 
plt.title('Restaurant types',fontsize = 20 ,pad=20)


# # 8. Distribution of restaurants according to approx cost for two people

# In[ ]:


fig, ax = plt.subplots(figsize=[15,7])
sns.distplot(zomato_en['approx_cost_for_2_people'],color="magenta")
ax.set_title('Approx cost for two people distribution',size=20,pad=15)
plt.xlabel('Approx cost for two people',size = 15)
plt.ylabel('Percentage of restaurants',size = 15)


# # 9. Showing True rate vs Predicted rate

# In[ ]:


plt.figure(figsize=(12,7))
preds_rf = RForest.predict(x_test)
plt.scatter(y_test,x_test.iloc[:,2],color="red")
plt.title("True rate vs Predicted rate",size=20,pad=15)
plt.xlabel('Rating',size = 15)
plt.ylabel('Frequency',size = 15)
plt.scatter(preds_rf,x_test.iloc[:,2],color="blue")


# # 10. Biggest chain restaurant vs. Best rated chain restaurants

# In[ ]:


plt.xkcd(False)
fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)


#BIGGEST CHAIN RESTAURANTS (top 10)
names,count = [],[]
top_ten_by_numbers = zomato['name'].value_counts().to_frame()[:10]
for index,row in top_ten_by_numbers.iterrows():
    names.append(index)
    count.append(row.values[0])

# ax1.barh(names,count) 
sns.barplot(count, names, palette="Blues_d",ax=ax1)   #Seaborn ! :D
ax1.set_xlabel("No. of restaurants")
ax1.set_title("BIGGEST CHAIN RESTAURANTS. (TOP 10)")

#TOP 10 BEST CHAIN RESTAURANT (top 10)  (Rated 4.8 and above)
name_by_rate,count_by_rate=[],[]
top_ten_by_rate_and_number = zomato[zomato['rate']>=4.8]['name'].value_counts().to_frame()[:10]
for index,row in top_ten_by_rate_and_number.iterrows():
    name_by_rate.append(index)
    count_by_rate.append(row.values[0])

# ax2.barh(name_by_rate,count_by_rate)  
ax2 = sns.barplot(count_by_rate,name_by_rate,palette="Reds_d",ax=ax2) 
ax2.set_xlabel("No. of restaurants")
ax2.set_title("BEST RATED CHAIN RESTAURANTS. (TOP 10)")
plt.tight_layout()
plt.show()


# # 11. Restaurant rating distribution

# In[ ]:


plt.figure(figsize=(15,8))
rating = zomato['rate']
plt.hist(rating,bins=20,color="red")
plt.title('Restaurant rating distribution', size = 20, pad = 15)
plt.xlabel('Rating',size = 15)
plt.ylabel('No. of restaurants',size = 15)


# # 12. Approx cost for 2 people distribution

# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(zomato.approx_cost_for_2_people)
plt.title('Approx cost for 2 people distribution', size = 20, pad = 15)
plt.xlabel('Approx cost for 2 people',size = 15)
plt.ylabel('Density',size = 15)


# # 13. Most popular cuisines of Bangalore

# In[ ]:


plt.figure(figsize=(15,8))
cuisines=zomato['cuisines'].value_counts()[:15]
sns.barplot(cuisines,cuisines.index)
plt.title('Most popular cuisines of Bangalore', size = 20, pad = 15)
plt.xlabel('No. of restaurants',size = 15)


# # 14. To see, Restaurant allows or doesn't allow, booking or online ordering.

# here we created a new column 'booking_ordering', where we got four conditions under which we see relationship between count of restaurants and ratings.

# In[ ]:


plt.xkcd(False)
dummy_2 = zomato.copy()
#create a new column 'booking_ordering', which basically tells whether a restaurant allows or doesn't allow, booking or online ordering.
dummy_2['booking_ordering']='None'  

for index,row in dummy_2.iterrows():
    if row['online_order'] == "Yes" and row['book_table']=='Yes':
        dummy_2.at[index,'booking_ordering'] = 'booking Available, online Available'      
    elif row['online_order'] == "Yes" and row['book_table']=='No':
        dummy_2.at[index,'booking_ordering'] = 'booking Unavailable, online Available'
    elif row['online_order'] == "No" and row['book_table']=='Yes':
        dummy_2.at[index,'booking_ordering'] = 'booking Available, online Unavailable'
    elif row['online_order'] == "No" and row['book_table']=='No':
        dummy_2.at[index,'booking_ordering'] = 'booking Unavailable, online Unavailable'

dummy_3 = dummy_2[['booking_ordering','rate']]

#Plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,1,1)
sns.countplot(data=dummy_3,x='rate',hue='booking_ordering',ax =ax1,palette='tab10')
plt.legend(loc='upper left')
plt.setp(ax1.get_legend().get_texts(), fontsize='22') # for legend text
plt.setp(ax1.get_legend().get_title(), fontsize='22') # for legend title
plt.tight_layout()
plt.show()


# In[ ]:





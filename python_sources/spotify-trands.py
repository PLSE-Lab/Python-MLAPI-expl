#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from scipy import stats
import squarify as sq
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix , classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path = '../input/top50spotify2019/top50.csv'
df = pd.read_csv(path , encoding='ISO-8859-1')
df.head()


# In[ ]:


df.shape


# In[ ]:


#Renaming the columns
df.rename(columns={'Track.Name':'track_name','Artist.Name':'artist_name','Beats.Per.Minute':'beats_per_minute','Loudness..dB..':'Loudness(dB)','Valence.':'Valence','Length.':'Length', 'Acousticness..':'Acousticness','Speechiness.':'Speechiness'},inplace=True)
df.head()


# In[ ]:


df.dtypes


# In[ ]:


# print(type(df['Genre']))
popular_genre = df.groupby('Genre').size()
print(popular_genre)
genre_list = df['Genre'].values.tolist()


# In[ ]:


print(df.groupby('artist_name').size())
popular_artist = df.groupby('artist_name').size()
print(popular_artist)
artist_list = df['artist_name'].values.tolist()


# In[ ]:


df.isnull().sum()


# In[ ]:


pd.set_option('precision',3)
df.describe()


# In[ ]:


#Finding out the skew for each attribute
skew=df.skew()
print(skew)
# Removing the skew by using the boxcox transformations
transform=np.asarray(df[['Liveness']].values)
df_transform = stats.boxcox(transform)[0]
# Plotting a histogram to show the difference 
plt.hist(df['Liveness'],bins=10) #original data
plt.show()
plt.hist(df_transform,bins=10) #corrected skew data
plt.show()


# In[ ]:


sns.boxplot( y = df['Popularity'])

fig , ax = plt.subplots(1,3)
fig.subplots_adjust(hspace = 0.6 ,wspace = 0.6)

sns.boxplot(y = df['beats_per_minute'], ax = ax[0])
sns.boxplot(y = df['Energy'],ax = ax[1])
sns.boxplot(y = df['Danceability'],ax = ax[2])
fig.show()


# In[ ]:


# each in denote 5 integers

ax = df['beats_per_minute'].plot.hist(bins = 20,alpha = 0.5)
plt.title('Histogram of Beats per minute')
plt.xlabel('beat Count')
plt.ylabel('Total')
plt.show()


# In[ ]:


fig , ax = plt.subplots(1,3)
fig.subplots_adjust(hspace = 0.6 , wspace = 0.6)

sns.boxplot( y = df['Loudness(dB)'] , ax = ax[0])
sns.boxplot( y = df['Liveness'], ax = ax[1])
sns.boxplot( y = df['Valence'],ax = ax[2])

fig.show()


# In[ ]:


fig , ax = plt.subplots(1,3)
fig.subplots_adjust(hspace = 0.8 , wspace = 0.8)

sns.boxplot(y = df['Length'], ax = ax[0])
sns.boxplot( y = df['Acousticness'],ax = ax[1])
sns.boxplot(y = df['Speechiness'],ax = ax[2])

fig.show()w


# In[ ]:


sns.catplot(y = 'Genre' , kind = 'count',
           palette = 'pastel' , edgecolor = '.6',
           data = df)


# In[ ]:


sns.catplot(x = 'Popularity' , y = 'Genre',kind = 'bar',
           palette = 'pastel',edgecolor = '.6',
           data = df)


# In[ ]:


import plotly.express as px

# Grouping it by Genre and track

plot_data = df.groupby(['Genre','track_name'] , as_index = False).artist_name.sum()

fig = px.line_polar(plot_data , theta = 'Genre', r ='artist_name',color = 'track_name')
fig.update_layout(title_text = 'James ingram and roxette',
                 height = 500 , width = 1000)
fig.show()


# In[ ]:


# Grouping it by Genre and artist

plot_data = df.groupby(['Genre' , 'track_name'],as_index = False).artist_name.sum()

fig = px.line(plot_data , x = 'Genre',y = 'artist_name',color = 'track_name')
fig.update_layout(title_text = 'james ingram and roxette',
                 height = 500 , width = 1000)

fig.show()


# In[ ]:


fig , ax = plt.subplots(1,3)
fig.subplots_adjust(hspace = 0.6 , wspace = 0.6)

sns.boxplot( y = df['BPM'],ax = ax[0])
sns.boxplot( y = df['Energy'],ax = ax[1])
sns.boxplot( y = df['Danceability'] , ax = ax[2])
fig.show()


# In[ ]:


transform1 = np.asarray(df[['Popularity']].values)
df_transform1 = stats.boxcox(transform1)[0]
# Plotting a histogram to show the difference 
# plt.hist(df['Popularity'],bins=10) original data
# plt.show()
# plt.hist(df_transform1,bins=10) #corrected skew data
# plt.show()
sns.distplot(df['Popularity'],bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='yellow')
plt.show()
sns.distplot(df_transform1,bins=10,kde=True,kde_kws={"color": "k", "lw": 2, "label": "KDE"},color='black') #corrected skew data
plt.show()


# In[ ]:


pd.set_option('display.width',100)
pd.set_option('precision',3)
correlation = df.corr(method = 'spearman')
plt.hist(correlation)


# In[ ]:


# Bar graph to see the number of songs of each genre

fig , ax = plt.subplots(figsize = (30,12))
length = np.arange(len(popular_genre))
plt.bar(length , popular_genre , color = 'red',edgecolor = 'black' , alpha = 0.7)
plt.xticks(length,genre_list)
plt.title('Most popular genre',fontsize = 18)
plt.xlabel('Genre',fontsize = 16)
plt.ylabel('Number of songs',fontsize = 16)
plt.show()


# In[ ]:


# Heatmap of the correlation

plt.figure(figsize = (10,10))
plt.title('Correlation heatmap')
sns.heatmap(correlation,annot=True , vmin = -1,vmax = 1 , cmap = 'GnBu_r',center = 1)


# In[ ]:


fig , ax = plt.subplots(figsize = (12,12))
length = np.arange(len(popular_artist))
plt.barh(length , popular_artist , color = 'blue',edgecolor = 'black',alpha = 0.7)
plt.yticks(length , artist_list)
plt.title('Most Popular artist',fontsize = 18)
plt.ylabel('artist',fontsize = 16)
plt.xlabel('Number of songs',fontsize = 16)
plt.show()


# In[ ]:


# Analysing the reletionship between energy and loudness

fig = plt.subplots(figsize = (10,10))
sns.regplot(x = 'Energy' , y = 'Loudness(dB)',data = df , color = 'red')


# In[ ]:


fig = plt.subplots(figsize=(10,10))
plt.title('Dependence Between energy and popularity')
sns.regplot(x = 'Energy' , y = 'Popularity',ci = None , data = df)
sns.kdeplot(df.Energy , df.Popularity, shade=True,cmap="Purples_d")


# In[ ]:


scatter_matrix(df)
plt.gcf().set_size_inches(30,30)
plt.show()


# In[ ]:


df.plot(kind = 'box' , subplots = True)
plt.gcf().set_size_inches(30,30)
plt.show()


# In[ ]:


plt.figure(figsize = (14,8))
sq.plot(sizes = df.Genre.value_counts(),label = df["Genre"].unique(),alpha = .8)
plt.axis('off')
plt.show()


# In[ ]:


#Pie charts 
labels = df.artist_name.value_counts().index
sizes = df.artist_name.value_counts().values
colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','cyan', 'green', 'black','yellow']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, colors=colors)
autopct=('%1.1f%%')
plt.axis('equal')
plt.show()


# In[ ]:


# Linear regression , frist create test and train dataset

x = df.loc[:,['Energy','Danceability']].values
y = df.loc[:,'Popularity'].values


# In[ ]:


# Creating a test training dataset

X_train , X_test , y_train ,y_test = train_test_split(x,y, test_size = 0.30,random_state = 42)


# In[ ]:


# Linear Regression

regressor = LinearRegression()
regressor.fit(X_train , y_train)
print(regressor.intercept_)
print(regressor.coef_)


# In[ ]:


# Displaying the diffrence btween the actual and the predicted

y_pred = regressor.predict(X_test)
df_output = pd.DataFrame({'Actual' : y_test , 'Predicted':y_pred})
print(df_output)


# In[ ]:


# Checking the accuracy of linear Regression

print('Mean Absoulute Error:',metrics.mean_absolute_error(y_test , y_pred))

print('Mean Squared Error:',metrics.mean_squared_error(y_test , y_pred))
print('Root mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[ ]:


plt.figure(figsize = (10,10))
plt.plot(y_pred , y_test , color = 'black',linestyle = 'dashed',marker = '*',markerfacecolor = 'red',markersize = 10)
plt.title('Error analysis')
plt.xlabel('Predicted values')
plt.ylabel('Test values')


# In[ ]:


# Cross validation score

x = df.loc[:,['Energy','Danceability']].values
y = df.loc[:,['Popularity']].values

regressor = LinearRegression()
mse = cross_val_score(regressor , X_train , y_train,
                     scoring = 'neg_mean_squared_error',cv = 5)
mse_mean = np.mean(mse)
print(mse_mean)
diff = metrics.mean_squared_error(y_test , y_pred) - abs(mse_mean)
print(diff)


# In[ ]:


x = df.loc[:,['artist_name']].values
y = df.loc[:,'Genre'].values


# In[ ]:


# Label Encodding of features

encoder = LabelEncoder()
x = encoder.fit_transform(x)
x = pd.DataFrame(x)


# In[ ]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

# Scaling to Between 0,1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


# KNN Classifabscation
# Sorted(sklearn.neighbors.valid_metric)

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(x_train , y_train)
y_pred = knn.predict(x_test)


# In[ ]:


error = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train , y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize = (10,10))
plt.plot(range(1,30),error , color = 'black',marker = 'o',markerfacecolor = 'cyan',markersize = 10)
plt.title('Error rate K value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[ ]:


sns.boxplot( y = raw_data['Popularity'])


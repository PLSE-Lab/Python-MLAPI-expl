#!/usr/bin/env python
# coding: utf-8

# **Introduction **
# 
# Are you interested on:
# 1. Which YouTube channel have most viewers, subscribers, video uploads etc?
# 2. Does more video upload gives out the more video views and more subscribers?
# 3. Does more subscriber gives more video views?
# 4. Is there a way to predict the number of subscribers based on the number of video uploaded by the channel and number of video views on it? 
# 
# Then, here I have tried to answer some of those questions using some visual tools and some analytical tools. 

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm


# In[ ]:


df = pd.read_csv('../input/data.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


#df['Subscribers'] = df['Subscribers'].convert_objects(convert_numeric=True)
#df['Video Uploads'] = df['Video Uploads'].convert_objects(convert_numeric=True)

df['Subscribers'] = pd.to_numeric(df['Subscribers'], errors='coerce')
df['Video Uploads'] = pd.to_numeric(df['Video Uploads'], errors='coerce')


# **Exploratory data analysis**
# 
# Here I start with plotting some bar graphs showing top 20 in each kind of classification of the channels. First three are top 20 by their ranking, where their number of viewers, subscribers and video views are presented. The second three are top 20 based on each of the group themselves. 

# In[ ]:


df.head(20).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Number of subscribers of top 20 channels')


# In[ ]:


df.head(20).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Number of video views of top 20 channels')


# In[ ]:


df.head(20).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Number of video uploads of top 20 channels')


# In[ ]:


df.sort_values(by = ['Subscribers'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Subscribers')
plt.title('Top 20 channels with maximum number of subscribers')


# In[ ]:


df.sort_values(by = ['Video views'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Video views')
plt.title('Top 20 channels with maximum number of video views')


# In[ ]:


df.sort_values(by = ['Video Uploads'], ascending = False).head(20).plot.bar(x = 'Channel name', y = 'Video Uploads')
plt.title('Top 20 channels with maximum number of video uploads')


# Here I am interested how all the channels in the list distribute in terms of subscribers, video uploads and subscribers going from maximum to minimum in each class. Interestingly there is huge peak at the top list and tend to gain a plateau for the other channels quickly. 

# In[ ]:


df.sort_values(by = ['Subscribers'], ascending = False).plot(x = 'Channel name', y = 'Subscribers')
plt.xlabel('Ranking by subscribers')
plt.ylabel('Number of subscribers')


# In[ ]:


df.sort_values(by = ['Video views'], ascending = False).plot(x = 'Channel name', y = 'Video views')
plt.xlabel('Ranking by video views')
plt.ylabel('Number of video views')


# In[ ]:


df.sort_values(by = ['Video Uploads'], ascending = False).plot(x = 'Channel name', y = 'Video Uploads')
plt.xlabel('Ranking by video uploads')
plt.ylabel('Number of video uploads')


# **Analysing by channel grades**

# In[ ]:


grade_name = list(set(df['Grade']))
grade_name


# In[ ]:


df_by_grade = df.set_index(df['Grade'])

count_grade = list()
for grade in grade_name:
    count_grade.append(len(df_by_grade.loc[[grade]]))


# In[ ]:


df_by_grade.head()


# In[ ]:


print(count_grade)
print(grade_name)


# In[ ]:


grade_name[2] = 'missing'


# In[ ]:


labels = grade_name
sizes = count_grade

explode1 = (0.2, 0.2, 0.5, 0.2, 0.2, 0.2)
color_list = ['green',  'red', 'gold', 'blue', 'lightskyblue', 'brown']

patches, texts = plt.pie(sizes, colors = color_list, explode = explode1, 
                         shadow = False, startangle = 90, radius = 3)
plt.legend(patches, labels, loc = "best")
plt.axis('equal')
plt.title('Classification of channels by grades')
plt.show()


# In[ ]:


df.describe()


# In[ ]:


props = dict(boxes="gold", whiskers="Black", medians="Black", caps="Black")
df.plot.box(color=props, patch_artist=True)
plt.yscale('log')
plt.ylabel('Log count')


# **Relation between variables**
# 
# Looking at the plot below, it is seen that number of subscribers is positively correlated with the number of viewers. That is expected. But the number of subscribers is negativley correlated with the number of video uploaded by that channel. This might be surprising. The video channels attracting the larger number of viwers and subscribers are uploading smaller number of videos. 
# 
# 

# In[ ]:


plt.subplots(figsize=(8, 5))
sns.heatmap(df.corr(), cmap = 'RdGy')
plt.title('Correlation Matrix Plot')


# The data contains non numeric values. So if the cleaned data is presented on the correlation scatter plot matrix the above mentioned conclusion about the correlation of three variables is more evident. 

# In[ ]:


df_clean = df.dropna()


# In[ ]:


sns.pairplot(df_clean)


# **Linear model**
# 
# Here I tried to make a linear model based on the data. I am tring to predict the number of subscribers given the 
# the number of video uploaded and number of video viewed. First started with the linear relation between two variables. 

# In[ ]:


X = df_clean[['Video Uploads', 'Video views']]
Y = df_clean[['Subscribers']]


# 20% of the data is randomly splitted for the testing purpose. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


# In[ ]:


lm = LinearRegression()
lm.fit(X_train.dropna(),y_train.dropna())


# In[ ]:


predictions = lm.predict(X_test)


# It is seen that there is already good correlation between the predicted value of the number of subscribers and the observed number of them in the test set. So the model is working satisfactorily for the data it never seen in the training. 

# In[ ]:


plt.scatter(y_test,predictions, color = 'red')
plt.xlabel('Y in test set')
plt.ylabel('Predicted Y')


# In[ ]:


sns.residplot(y_test, predictions,  color="g")
plt.ylabel('d')
plt.xlabel('instances')
plt.title('standardized residual plot')


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


coefficients = pd.DataFrame(X.columns)
coefficients['coefficients']=lm.coef_[0]
coefficients


# In[ ]:


model = sm.OLS(Y, X).fit() 
predictions = model.predict(X_test)


# In[ ]:


model.summary()


# **Working over the skewness of the data**
# 
# Form the following three histogram, we can see that all three variables are highly positively skewed. 

# In[ ]:


df['Subscribers'].hist(bins = 200)
plt.xlabel('Number of subscribers')
plt.ylabel('Number of channels')


# In[ ]:


df['Video views'].hist(bins = 200)
plt.xlabel('Number of video views')
plt.ylabel('Number of channels')


# In[ ]:


df['Video Uploads'].hist(bins = 200)
plt.xlabel('Number of video uploads')
plt.ylabel('Number of channels')


# **Log transformation**
# 
# In view of the positive skewness of the data, simple log transformation could be a good choice to deal with.

# In[ ]:


np.log(df['Subscribers']).hist(bins = 20)
plt.xlabel('Log of number of subscribers')
plt.ylabel('Number of channels')


# In[ ]:


np.log(df['Video views']).hist(bins = 20)
plt.xlabel('Log of number of video views')
plt.ylabel('Number of channels')


# In[ ]:


np.log(df['Video Uploads']).hist(bins= 20)
plt.xlabel('Log of number of video uploads')
plt.ylabel('Number of channels')


# In[ ]:


df_log = pd.DataFrame()
df_log['Video_uploads_log'] = np.log(df_clean['Video Uploads'])
df_log['Video_views_log'] = np.log(df_clean['Video views'])
df_log['Subscribers_log'] = np.log(df_clean['Subscribers'])


# In[ ]:


df_log.head()


# In[ ]:


df_log.tail()


# **Study of correlation with log transformation**

# In[ ]:


plt.subplots(figsize=(8, 5))
sns.heatmap(df_log.corr(), cmap = 'RdGy')


# From the above correlation plot the correlation coefficient of the variables have not been changed after the log transformation. At least the positive correlation remains the positive and vice versa. 
# 
# But if we look at the scatter plot below, visually the negative correlation between video uploads and subscribers seem to have gone. This is the effect of log transformation which is not to be confued thinking they have positive correlations.

# In[ ]:


sns.pairplot(df_log)


# **Linear model with log transformation**

# In[ ]:


X2 = df_log[['Video_uploads_log', 'Video_views_log']]
Y2 = df_log[['Subscribers_log']]


# In[ ]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, Y2, test_size = 0.2)


# In[ ]:


lm2 = LinearRegression()
lm2.fit(X2_train.dropna(),y2_train.dropna())


# In[ ]:


predictions2 = lm2.predict(X2_test)


# In[ ]:


plt.scatter(y2_test,predictions2, color = 'red')
plt.xlabel('Y in test set')
plt.ylabel('Predicted Y')


# In[ ]:


sns.residplot(y2_test, predictions2,  color="g")
plt.ylabel('d')
plt.xlabel('instances')
plt.title('standardized residual plot')


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y2_test, predictions2))
print('MSE:', metrics.mean_squared_error(y2_test, predictions2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, predictions2)))


# In[ ]:


coefficients2 = pd.DataFrame(X2.columns)
coefficients2['coefficients']=lm2.coef_[0]
coefficients2


# In[ ]:


model2 = sm.OLS(Y2, X2).fit() 
predictions2 = model2.predict(X2_test)


# In[ ]:


model2.summary()


# **Comparing the result with and without log transformation**
# 
# Without using log: 
# 
# Y = a X_1  + b X_2 + c 
# 
# With log 
# 
# ln(Y) = p ln(X_1) + q ln(X_2) + r
# 
# From the later 
# 
# Y = exp( p .... ) = X_1 ^ p + X_2 ^ q + e^r 

# In the following, the prediction made by the log transformation is compared with the one done directly.  The relation is mentioned in above shell.

# In[ ]:


p = coefficients2['coefficients'][0]
q = coefficients2['coefficients'][1]


# In[ ]:


def pred_from_log(x, y):
    return x ** p + y ** q


# In[ ]:


X_test.head()


# In[ ]:


vid_upl_test = np.array(X_test['Video Uploads'])
vid_viw_test = np.array(X_test['Video views'])


# In[ ]:


prediction_log = pred_from_log(vid_upl_test, vid_viw_test)


# It is nice that both predictions are highly correleted.

# In[ ]:


plt.scatter(predictions, prediction_log, color = 'r', alpha = 0.5)
plt.xlabel('prediction without log transformation')
plt.ylabel('prediction with log transformation')


# The direct plot of the difference shows that log transformation tend to predict higher value than that without log if anything. There is no way it can predict lower though. 

# In[ ]:


plt.scatter(range(len(X_test)), predictions - prediction_log, color = 'red', alpha = 0.5)
plt.xlabel('count of test data')
plt.ylabel('difference of prediction with and without log')


# **Conclusion**
# 
# Conclusion of the study is the following:
# * The number of subscribers is proportional to the number of views. 
# * The number of subscribers in negatively correlated witht the number of video uploads by the channel. 
# * Linear model was tested for prediction of number of subscriber as a function of number of video uploads and number of video views.
# *  Log transformation on the linear model gives the one sided biased prediction in comparison to the one without such transformation. 
# 

# In[ ]:





# In[ ]:





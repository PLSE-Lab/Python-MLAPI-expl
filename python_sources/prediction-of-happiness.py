#!/usr/bin/env python
# coding: utf-8

# ## Load the Data

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df_2015 = pd.read_csv("../input/2015.csv");


# ## Dataset Cleansing
# 
# 

# #### Combine three datasets into one and create new column Year

# In[ ]:


df_2015.drop('Standard Error', axis=1, inplace=True)


# In[ ]:


df_2015.head()


# ## EDA

# In[ ]:


df = df_2015[['Country', 'Happiness Score']].sort_values('Happiness Score', ascending=False).nlargest(10, 'Happiness Score')

df.plot.bar(x='Country',y='Happiness Score')


# ### Correlation heatmap 

# In[ ]:


import seaborn as sns


# In[ ]:


corr = df_2015.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(figsize=(10, 7))
corr = df_2015.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)
axes.set_title("2015")


# ### Faceted Plot
# packages and code: https://plot.ly/python/facet-plots/

# In[ ]:


get_ipython().system('pip -q install ggplot')


# In[ ]:



import ggplot
.../site-packages/ggplot/stats/smoothers.py
ggplot(df_2015, aes(x="Freedom", y="Happiness Score",color="Region",size = 'Happiness Score')) + geom_point(size=200)


# In[ ]:


from ggplot import *
ggplot(df_2015, aes(x="Health (Life Expectancy)", y="Happiness Score",color="Region",size = 'Happiness Score')) + geom_point(size=200)


# In[ ]:


from ggplot import *
ggplot(df_2015, aes(x="Trust (Government Corruption)", y="Happiness Score",
                               color="Region",size = 'Happiness Score')) + geom_point(size=200)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


g = sns.FacetGrid(df_2015) 
g.map(sns.regplot, "Family","Economy (GDP per Capita)")
g.add_legend()


# ###Pairplot

# In[ ]:


g = sns.pairplot(df_2015[['Happiness Score','Economy (GDP per Capita)','Family','Health (Life Expectancy)']],kind = "reg")


# ## Create Training and Test *Set*

# Drop the Happiness Rank from training datasets since it is perfectly correlated with Happiness Score.

# In[ ]:


df_2015 = df_2015.drop(columns = ['Happiness Rank'],axis = 1)


# In[ ]:


df_2015.head()


# ## Machine Learning Model 

# ### Linear Model

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split

feature_col_names=["Economy (GDP per Capita)",'Family','Health (Life Expectancy)','Freedom','Trust (Government Corruption)','Generosity','Dystopia Residual']
score=['Happiness Score']

X = df_2015[feature_col_names].values
y =df_2015[score].values
split_size=0.3

X_train, X_test,y_train,y_test= train_test_split(X,y ,test_size=split_size)

#training
mlregr=linear_model.LinearRegression()
mlregr.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#predicting
y_pred= mlregr.predict(X_test)

#for i in range(len(y_test)):
    #print ("Actual:" ,y_test[i] ," Predicted with linear regression:" ,y_pred[i])
    
print("Coefficients: ", mlregr.coef_)
print("Intercepts:  " , mlregr.intercept_)
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred))
#


# In[ ]:


plt.scatter(y_test,y_test)
plt.plot(y_test,y_pred,linewidth=1)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# ###SVM
# 

# In[ ]:


from sklearn import svm

svr= svm.SVR(gamma='scale')
svr.fit(X,y.ravel())


# In[ ]:


y_pred_svr=svr.predict(X_test)


# In[ ]:


print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred_svr))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,y_pred_svr))


# In[ ]:


#Plotting Predicted vs Actual Happiness Score

plt.scatter(y_test,y_test)
plt.plot(y_test,y_pred_svr)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


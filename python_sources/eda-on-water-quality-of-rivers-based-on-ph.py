#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyforest')


# In[ ]:


from pyforest import *


# In[ ]:


df=pd.read_csv("../input/water-quality-data/waterquality.csv", sep=',', engine='python')


# In[ ]:


df


# In[ ]:


df.isnull().sum()


# In[ ]:


df= df.fillna(df.mean())


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# Plotting a correlation matrix to see the relation between different features
# 
# Here positive values mean positively correlated i.e, if one increase then other also increases and vice versa

# In[ ]:


sns.heatmap(df.corr(),annot=True)


# Pairplot of df

# In[ ]:


df= df.drop("STATION CODE", axis=1)


# In[ ]:


df.head()


# In[ ]:


sns.pairplot(df)


# # Separating the pH column and using q values for the column separately

# ## The pH of river water is the measure of how acidic or basic the water is on a scale of 0-14. It is a measure of hydrogen ion concentration.The optimum pH for river water is around 7.4.
# 

# In[ ]:


ph= df["pH"]


# In[ ]:


ph.value_counts()


# In[ ]:


PH= pd.DataFrame(ph, index=None)


# In[ ]:


PH.head()


# In[ ]:


PH.pH.value_counts()


# generating the q values by using a specific range from the q table

# In[ ]:


PH["QI"]=PH.replace(to_replace =6.4,  
                            value =54)


# In[ ]:


PH.head()


# In[ ]:


PH["QI"]=PH["QI"].replace(to_replace =[6.5,6.7,6.8,6.9],  
                            value =75)

PH["QI"]=PH["QI"].replace(to_replace =[7.0,7.1,7.2,7.3,7.4],  
                            value =80)

PH["QI"]=PH["QI"].replace(to_replace =[7.5,7.6,7.7,7.8,7.9],  
                            value =95)

PH["QI"]=PH["QI"].replace(to_replace =[8.0,8.1,8.2,8.3,8.4],  
                            value =85)

PH["QI"]=PH["QI"].replace(to_replace =[8.5,8.6,8.7,8.8,8.9],  
                            value =65)

PH["QI"]=PH["QI"].replace(to_replace =[9.0,9.1,9.2,9.3,9.4],  
                            value =48)

PH["QI"]=PH["QI"].replace(to_replace =[9.5,9.6,9.7,9.8,9.9],  
                            value =30)

PH["QI"]=PH["QI"].replace(to_replace =[10.0,10.1,10.2,10.3,10.4],  
                            value =20)

PH["QI"]=PH["QI"].replace(to_replace =[10.5,10.6,10.7,10.8,10.9],  
                            value =12)

PH["QI"]=PH["QI"].replace(to_replace =[11.0,11.1,11.2,11.3,11.4],  
                            value =8)

PH["QI"]=PH["QI"].replace(to_replace =[11.5,11.6,11.7,11.8,11.9],  
                            value =4)

PH["QI"]=PH["QI"].replace(to_replace =[12.0,12.1,12.2,12.3,12.4,12.5,12.6,12.7,12.8,12.9,13.0,13.1,13.2,13.3,13.4,13.5,13.6,13.7,13.8],  
                            value =75)


# In[ ]:


PH["QI"].value_counts()


# In[ ]:


PH.head(), PH.tail()


# In[ ]:


sns.jointplot(x="pH", y="QI", data=PH, kind="reg")


# From above we can see that maximum of the river water pH lies b/w 6.5 and 8.5 and QI lies b/w 80 and 100

# In[ ]:


sns.distplot(PH["pH"], kde=True, bins=10)


# concatenating qi values from ph with state

# In[ ]:


ls= df[["LOCATIONS","STATE"]]


# In[ ]:


ls.head()


# In[ ]:


df_col_merged = pd.concat([ls, PH], axis=1)


# In[ ]:


df_col_merged.head()


# A barplot to show the relation of QI and States

# In[ ]:


ax=sns.barplot(x="STATE", y= "QI", data=df_col_merged)
ax.set_xticklabels(ax.get_xticklabels(), rotation=65, horizontalalignment='right')


# # More work is to be done in the notebook and will update the whole notebook soon

# In[ ]:


X= df_col_merged.drop(["LOCATIONS","STATE","QI"], axis=1)


# In[ ]:


X.head()


# In[ ]:


X.shape


# In[ ]:


y= df_col_merged["pH"]


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# In[ ]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


ridge= Ridge()


# In[ ]:


parameters= {"alpha":[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,40,50,60,70,80,90,100]}

ridge_regressor= GridSearchCV(ridge,parameters,scoring="neg_mean_squared_error", cv=10)

ridge_regressor.fit(X,y)


# In[ ]:


ridge_regressor.best_params_


# In[ ]:


ridge_regressor.best_score_, ridge_regressor.score


# In[ ]:


ridge_regressor.cv_results_


# In[ ]:


df2= pd.DataFrame(ridge_regressor.cv_results_)


# In[ ]:


df2


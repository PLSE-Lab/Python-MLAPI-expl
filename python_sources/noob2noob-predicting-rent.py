#!/usr/bin/env python
# coding: utf-8

# # 0 - Introduction
# Heyy guys, my name is Gabriel Riquieri and I'm a computer science student. I started to study data science a time ago and realy like it and start to work in some projects. That one it's the first project that i share. I decided to do it in a didactic way to solidify my own knowledge.  I'm not a professor or some Phd student, so my code and explanations may contain some mistakes or maybe there is some way do do with a better performace. I hope you guys enjoy this small work, feel free to leave a comment or email me in gabriel.riquieri@ufu.br

# # 1 - Importing dataframe 
# If you are some kind a moster that skip the introductions, I undestand and here we go!
# 
# First things first, import pandas to work with dataframes and them read the .csv file with the data

# In[ ]:


import pandas as pd


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


csv_path = '/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv'
df = pd.read_csv(csv_path)


# In[ ]:


df.head(3)


# In[ ]:


df.shape


# # 2 - 'What do we have and waht do we want?'

# We have almost 11000 rows and 13 features, we have a lot of data and before start to plat with this deta let's define what do we wanto to predict. In this case i will be trying to predict the 'rent amount'; So we have 12 independent variables and 1 dependent variable(target). Our target is a price, so it's a continuous numeric variable, then we can conclude that we have a regression problem

# # 3 - Exploratory data analysis(EDA) and visualization

# Our variables are numerical or categorical?

# In[ ]:


df.info()


# So we have 4 categorical features and 8 numerical features. That nice because, in a short future, when we will be building models, we will need that this data be all numeric

# ## 3.1 - Checking and dealing with missing values

# Now that we know what we have, let's start to search for what we don't have(wow, that's sound very philosophical), let's try to find missing values

# In[ ]:


miss = df.isnull().sum()
miss_= (df.isnull().sum()/df.isnull().count())*100

missing_data = pd.concat([miss, miss_], axis=1, keys=['Total', 'Percent'])
missing_data


# And that is... strange.. what is a problem without variables missing? Let's keep searching

# In[ ]:


df['city'].value_counts()


# In[ ]:


df['floor'].value_counts()


# Enemy spooted!
# 
# '-' doesn't seem like a value to enumerate floors.. and I'm pretty sure that we don't have a building with 301 floors in those cities. Let's transform those values to '0', tranform these feature for numerical, get the median and replace those zeros

# In[ ]:


df['floor'] = df['floor'].replace(['-','301'], 0)
df['floor'].value_counts()


# In[ ]:


df['floor'] = df['floor'].astype(int)


# In[ ]:


df.info()


# In[ ]:


df['floor'] = df['floor'].replace(0, df['floor'].median())
df['floor'].value_counts()


# Good work!

# ## 3.2 - Visualization
# We can use plots to have and idea of the way that our data are distributed. It's simple and effective.

# In[ ]:


ax = df['city'].value_counts().plot(kind='bar', figsize=(15,6))
ax.set_xlabel('Cities')
ax.set_title('Cities visualization', fontsize = 25)


# In[ ]:


ax = df['rooms'].value_counts().plot(kind='bar', figsize=(20,6))
ax.set_xlabel('Number of rooms')
ax.set_title('Rooms visualization')


# In[ ]:


ax = df['bathroom'].value_counts().plot(kind='bar', figsize=(15,6))
ax.set_xlabel('Number of bathroom')
ax.set_title('Bathroom visualization', fontsize = 25)


# ## 3.3 - Turning categorical into numerical
# As I said before, our model work with numeric values. We need to turn our categorical features into numerical features; We could use ``pd.get_dummies()`` but because we only have two possibles categories in each feature, we can simply use ``replace()`` 
# 
# * I will discard 'city' that why i'm not work with this feature; If our model get a poor accuracy we can go back here and transform 'city' values in numerical values too

# ##### Animal

# In[ ]:


df['animal'].value_counts()


# * acept = 1
# * not acept = 0

# In[ ]:


df['animal'] = df['animal'].replace(['acept'], 1)
df['animal'] = df['animal'].replace(['not acept'], 0)


# ##### Furniture

# In[ ]:


df['furniture'].value_counts()


# * furnished = 1
# * not furnished = 0

# In[ ]:


df['furniture'] = df['furniture'].replace(['furnished'], 1)
df['furniture'] = df['furniture'].replace(['not furnished'], 0)


# In[ ]:


df.head()


# Nice! Works weel

# # 3.4 - Correlation

# Now let's study the correlation between our features. To do that we can plor a correalation table but I think that is better visualize in a scatter plot

# In[ ]:


df.corr()


# ### City

# In[ ]:


import matplotlib.pyplot as plt

x= df['city']
y= df['rent amount (R$)']

plt.scatter(x,y)


# ### Area

# In[ ]:


import matplotlib.pyplot as plt

x= df['area']
y= df['rent amount (R$)']

plt.scatter(x,y)


# 'Enemy outliar spotter! What your orders Sir?'
# 
# 'Drop then all!'

# In[ ]:


df.sort_values(by = 'area', ascending = False)[:2]


# In[ ]:


df = df.drop(df[df['area'] == 46335].index)
df = df.drop(df[df['area'] == 12732].index)


# In[ ]:


import matplotlib.pyplot as plt

x= df['area']
y= df['total (R$)']

plt.scatter(x,y)


# We still have some enemys on the field, but I will let then live another day...

# ### (Highest in the )Rooms
# Hope you catch the reference ;)

# In[ ]:


x= df['rooms']
y= df['rent amount (R$)']

plt.scatter(x,y)


# ### Bathroom

# In[ ]:


x= df['bathroom']
y= df['rent amount (R$)']

plt.scatter(x,y)


# ### Parking spaces

# In[ ]:


x= df['parking spaces']
y= df['rent amount (R$)']

plt.scatter(x,y)


# ### HOA

# In[ ]:


import matplotlib.pyplot as plt

x= df['hoa (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)


# ### Total 

# In[ ]:


y= df['rent amount (R$)']
x= df['total (R$)']

plt.scatter(x,y)


# ### Property tax

# In[ ]:


import matplotlib.pyplot as plt

x= df['property tax (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)


# 'Enemy spooted, you know what to do...'

# In[ ]:


df.sort_values(by = 'property tax (R$)', ascending = False)[:2]


# In[ ]:


df = df.drop(df[df['property tax (R$)'] == 10830].index)


# In[ ]:


x= df['property tax (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)


# Much better!

# ### Fire insurance
# Come on, keep going, you're on fire! 
# (Ok.. that has terrible...)

# In[ ]:


x= df['fire insurance (R$)']
y= df['rent amount (R$)']

plt.scatter(x,y)


# # 4 - Feature selection
# In this case, we have only a few features, so it's not a problem use then all to create the model. But maybe you guys can know, we have some huge datasets, some with much more then 20 features, so here we will see how to select some good predict features

# In[ ]:


x = df.drop(['total (R$)','city','rent amount (R$)'], axis=1)
y = df['rent amount (R$)']


# A good way to select a variable is using chi^2. I realy recomend that you guys search about this method,  it have a lot of math and I will not spend so much time here showing you guys how the algorithm works, but in resume, we are searching for a feature with large chi^2 value

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


algorithm = SelectKBest(score_func=chi2, k=5)

best_features = algorithm.fit_transform(x,y)


# In[ ]:


print('scores:',algorithm.scores_)
#print('Resultado da transformacao:\n',dados_das_melhores_preditoras)


# In[ ]:


chi_scores = algorithm.scores_


# In[ ]:


chi_ = pd.DataFrame(data=chi_scores,columns = ['Chi^2'], index=['area','rooms','bathroom','parking spaces','floor','animal','furniture','hoa (R$)','property tax (R$)','fire insurance (R$)'])


# In[ ]:


chi_['Chi^2'].sort_values(ascending=False)


# So, if we would doesnt not wanted to use all features, we could use some of those features good ranked

# # 5 - Modelling

# Now it's time for the fun step, here it is the different kinds of regression algorithms that I will use:
# * Linear regression
# * Ridge regression
# * Lasso regression
# * Elastic net
# * Decision tree regressor

# Classic train and test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)


# ## 5.1 - Linear regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


#lm = linear model
lm = LinearRegression()


# In[ ]:


lm.fit(x_train, y_train)


# In[ ]:


a_lm =lm.score(x_test, y_test)
print('R^2=', a_lm)         


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


kfold = KFold(n_splits=5,shuffle=True)


# In[ ]:


b_lm = cross_val_score(lm,x,y,cv=kfold)
print(b_lm.mean())


# Easy peasy right? We created our model object, trained with the ``lm.fit()`` and then get the score; After that We did a cross validation to double check our model accuracy. If you understood how we build and train this model, understand the following ones will be easy

# ## 5.2 - Ridge regression

# In[ ]:


from sklearn.linear_model import Ridge
#rm = Ridge model
rm = Ridge()


# In[ ]:


rm.fit(x_train, y_train)


# In[ ]:


a_rm =rm.score(x_test, y_test)
print('R^2=',a_rm)


# In[ ]:


kfold = KFold(n_splits=5,shuffle=True)


# In[ ]:


b_rm = cross_val_score(rm,x,y,cv=kfold)
print(b_rm.mean())


# Much like the before model right? We could variate the 'alpha' value using GridSearchCV, but as we already got a good result we don't need to do it

# ## 5.3 - Lasso regression

# In[ ]:


from sklearn.linear_model import Lasso
#lassom = lasso model
lassom = Lasso(alpha=1000, max_iter=1000, tol=0.1)


# In[ ]:


lassom.fit(x_train, y_train)

a_lassom = lassom.score(x_test, y_test)
print('R^2=',a_lassom)


# In[ ]:


kfold = KFold(n_splits=5,shuffle=True)


# In[ ]:


b_lassom = cross_val_score(lassom,x,y,cv=kfold)
print(b_lassom.mean())


# ## 5.4 - ElasticNet

# In[ ]:


from sklearn.linear_model import ElasticNet
#em = elastic model
em = ElasticNet(alpha=1,max_iter=5000, l1_ratio=0.5, tol=0.2)


# In[ ]:


em.fit(x_train, y_train)

a_em = em.score(x_test, y_test)
print('R^2=',a_em)


# In[ ]:


kfold = KFold(n_splits=5,shuffle=True)


# In[ ]:


b_em = cross_val_score(em,x,y,cv=kfold)
print(b_em.mean())


# Again, again and again.. Hope you guys are understanding.. I'm not commenting too much now because it's much like the linear regression and the ridge regression. One more time, i could be variating parameters in the lasso regression and in the elasticnet, but we are getting good results without change then

# ## 5.5 - Decision tree regressor

# In[ ]:


import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor


# In[ ]:


min_splits = np.array([2,3,4,5,6,7])
max_lever = np.array([3,4,5,6,7,9,11])
algorithm = ['mse', 'friedman_mse', 'mae']

valores_grid = {'min_samples_split':min_splits,
                'max_depth':max_lever,
                'criterion':algorithm
               }


# In[ ]:


tm = DecisionTreeRegressor()


# In[ ]:


grid_tm = GridSearchCV(estimator=tm, param_grid=valores_grid, cv=5, n_jobs=-1)
grid_tm.fit(x,y)


# In[ ]:


print('R2:',grid_tm.best_score_)
print('Min to split:',grid_tm.best_estimator_.min_samples_split)
print('Max depth:',grid_tm.best_estimator_.max_depth)
print('Algorithm:',grid_tm.best_estimator_.criterion)


# Now we have something that worth variate. To do this variation with the parameters we can use GridSearchCV

# Finishing our modelling work, let's compare the accuracy values for each model(using train_test_split and cross validation)

# In[ ]:


score_df_data = {'Linear regression':[0.9870,0.9871],
        'Ridge regression':[0.9870,0.9872],
        'Lasso regression':[0.9828,0.9823],
        'ElasticNet':[0.9866,0.9866],
        'Decision tree regression':['Nan',0.9929]}



score_df = pd.DataFrame(data=score_df_data ,index=['Train_test','Cross_val'])


# In[ ]:


score_df


# Pretty!
# 
# All moodels have a realy good accuracy
# 
# 'If that's good?'
# 
# In the reality that give me goosebumps, because I start to think that something is wrong. 

# # 6 - Predicting new values

# One way to see if that huge accuracy are correct is inserting new values to see how the model react.
# 
# I did some tests and looks that indeed we have good prediction

# In[ ]:


a = input('AREA:\n')
b = input('ROOMS:\n')
c = input('BATHROOM:\n')
d = input('PARKING SPACES:\n')
e = input('FLOOR:\n')
f = input('ANIMAL 1-YES/0-NO:\n')
g = input('FORNITURE 1-YES/0-NO:\n')
h = input('HOA:\n')
i = input('PROPERTY TAX:\n')
j = input('FIRE INSURANCE:\n')


# In[ ]:


l = rm.predict([[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j)]])


# In[ ]:


print('The total value suggest is R$',l[0])


# I did some tests and looks that indeed we have decent prediction; What about you guys? What do you think about our results? We can do a lot more in EDA, what would you add? ;)
# 
# 
# Leave a comment!

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# 
# >Welcome to my kernel. We're performing here some data cleansing, transformation, EDA and finally a ML Linear regression model for Graduate admission dataset. 
# 
# ![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRAGHhGeSTlVOtL2cwPvfAz61HAr20IOOeFIS5ODM80_LP9j67b&usqp=CAU)

# >Pulling Graduation dataset here. 
# >Creating df_train and df_test dataframes

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df_test = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
print(df_train.count()) 

print(df_test.count())


# In[ ]:


df_train.columns


# >`Serial No` column is of no use for the model. Drop col from train and test sets

# In[ ]:


df_train.drop('Serial No.', axis=1, inplace=True)
df_test.drop('Serial No.', axis=1, inplace=True)

df_train.head()


# > I'm estimating chance of admit greater than 75% should be confirmed admissions. 
# > It doesn't make this a classification problem, this is purely for data exploration purpose. 

# In[ ]:


df_train['admission'] =  np.where(df_train['Chance of Admit '] >= 0.75, 1, 0)
df_train.head()


# ##       Lets do some data exploration

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams['figure.figsize'] = (8, 6)
# plt.rcParams['font.size'] = 14


# In[ ]:


# Pandas scatter plot
df_train.plot(kind='scatter', x='GRE Score', y='CGPA')


# In[ ]:


sns.scatterplot(x='GRE Score', y='CGPA', hue="admission", data=df_train)


# > Check correlation between input features against Chance of Admission.
# 
# > For further insight, marking 75% chance of admit on charts

# In[ ]:


feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP','LOR ', 'CGPA', 'Research']

# multiple scatter plots in Seaborn
g = sns.pairplot(df_train, x_vars=feature_cols, y_vars='Chance of Admit ', kind='reg')

for chts in g.axes[0]: 
    chts.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')


# > **GRE Score, TOEFL Score** and **CGPA** seems to be more correlated features

# In[ ]:


sns.relplot(x='GRE Score', y='CGPA',
                 col="University Rating", hue="admission", 
                 kind="scatter", data=df_train)


# In[ ]:


sns.relplot(x='GRE Score', y='TOEFL Score',
                 col="University Rating", hue='admission', 
                 kind="scatter", data=df_train)


# In[ ]:


pdf=df_train.groupby(['Research','University Rating']).mean().reset_index()
pdf


# In[ ]:


bg = sns.boxplot(y="Chance of Admit ",  x= 'Research', palette=["m", "g"], data=df_train)
sns.despine(offset=10, trim=True)
bg.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')


# In[ ]:


bg = sns.boxplot(y="Chance of Admit ",  x= 'University Rating', data=df_train)
sns.despine(offset=10, trim=True)
bg.axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')


# In[ ]:


gr = sns.catplot(x = "University Rating",   
            y = "Chance of Admit ",       
            hue = "Research",  
            data = df_train.groupby(['Research','University Rating']).mean().reset_index() , 
            kind = "bar")
gr.axes[0][0].axes.axhline(y= 0.75, linewidth=2, color='r', ls='--')


# In[ ]:


sns.relplot(x='CGPA', y='Research',
            col="University Rating", hue='admission', 
            kind="scatter", data=df_train)


# > lets check correlation among all variables using heatmap

# In[ ]:


colormap = sns.diverging_palette(100, 5, as_cmap=True)
sns.heatmap(df_train.corr(), annot = True, cmap= colormap, cbar=True,  fmt=".2f" )


# > correlation of variables, different heatmap visualization.
# 
# > ref: https://seaborn.pydata.org/generated/seaborn.heatmap.html

# In[ ]:


corr = df_train.corr()
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(100, 5, as_cmap=True)

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(8, 8))
    ax = sns.heatmap(corr,cmap=colormap,linewidths=.5, annot=True, mask=dropSelf )


# ># ** Linear regression model **

# In[ ]:


# import, instantiate, fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.model_selection import train_test_split


linreg = LinearRegression()


# In[ ]:


feature_cols = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

## training set
X = df_train[feature_cols]
y = df_train['Chance of Admit ']

## test set 
X_test = df_test[feature_cols]
y_test = df_test['Chance of Admit ']

## fit model
linreg.fit(X, y)


# > coefficients for each features

# In[ ]:


# print the coefficients
print(list(zip(feature_cols,linreg.coef_)))


# In[ ]:


# define a function that accepts a list of features and returns RMSE, prediction 
def train_test_rmse(feature_cols, X , y):
    y_pred = linreg.predict(X)
    return np.sqrt(mean_squared_error(y, y_pred)), y_pred


# In[ ]:


rmse, ypred = train_test_rmse(feature_cols, X_test , y_test)

df_test['admission_predict'] = y_pred
print(rmse)


# > Lets have a look at error as per MAE, MSE & RMSE 

# In[ ]:


print('MAE:',  mean_absolute_error(y_test, y_pred), ' ',  (1./len(y_test))*(sum(abs(y_test-y_pred))))
print('MSE:', mean_squared_error(y_test, y_pred), ' ',   (1./len(y_test))*(sum((y_test-y_pred)**2)))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)), ' ', sqrt((1./len(y_test))*(sum((y_test-y_pred)**2))))


# In[ ]:


fig, ax = plt.subplots()
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(7, 7)})
sns.regplot(x=y_test, y=y_pred,  scatter=False, ax=ax);


# In[ ]:


##Check for Linearity
f = plt.figure(figsize=(14,5))
## linear
ax = f.add_subplot(121)
sns.scatterplot(y_test,y_pred,ax=ax,color='r')
ax.set_title('Check for Linearity:\n Actual Vs Predicted value')

# Check for Residual error
f = plt.figure(figsize=(14,5))
ax = f.add_subplot(121)
sns.distplot((y_test-y_pred), bins = 50)
ax.axvline((y_test - y_pred).mean(),color='r',linestyle='--')
ax.set_title('Check for Residual normality & mean: \n Residual eror');


# In[ ]:


sns.distplot(y_test,hist=True,label = 'Actual')
sns.distplot(y_pred,hist=True, label ='Predicted')
plt.legend(loc="upper right")
plt.xlabel('Prediction')


# >Lets check if scaling inpute features does have any impact on model

# In[ ]:


from sklearn.preprocessing import StandardScaler as SS
ss = SS()
X_ss = ss.fit_transform(X)
ss1 = SS()
X_test_ss = ss.fit_transform(X_test)


# > There is no much improvement, for a simple model like linear regression won't have any improvement on scaling features

# In[ ]:


linreg.fit(X_ss, y)
 
rmse_ss = train_test_rmse(feature_cols, X_test_ss , y_test)[0]
y_pred_ss = train_test_rmse(feature_cols, X_test_ss , y_test)[1]
print(rmse_ss)


# In[ ]:


sns.distplot(y_test,hist=True,label = 'Actual')
sns.distplot(y_pred_ss,hist=True, label ='Predicted (for StandardScaler inputs)')
plt.legend(loc="upper right")
plt.xlabel('Prediction')


# ># Admission chance prediction done ! 

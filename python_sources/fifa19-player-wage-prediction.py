#!/usr/bin/env python
# coding: utf-8

# # Fifa19 - Player Wage Prediction

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/data.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.drop(columns=['Unnamed: 0'], inplace=True)


# In[ ]:


df.sample(5)


# In[ ]:


df.describe()


# ## Exploratory Analysis
# Wages of players to be predicted using their Age, Overall Rating, Potential, Value, Height, Weight, International Reputation and Position.

# ### Distribution of Overall Rating

# In[ ]:


sb.set_style('whitegrid')


# In[ ]:


bins = np.arange(df['Overall'].min(), df['Overall'].max()+1, 1)

plt.figure(figsize=[8,5])
plt.hist(df['Overall'], bins=bins)
plt.title('Overall Rating Distribution')
plt.xlabel('Mean Overall Rating')
plt.ylabel('Count')
plt.show()


# ### Age vs Overall Rating

# In[ ]:


plt.figure(figsize=[16,5])
plt.suptitle('Overall Rating Vs Age', fontsize=16)

plt.subplot(1,2,1)
bin_x = np.arange(df['Age'].min(), df['Age'].max()+1, 1)
bin_y = np.arange(df['Overall'].min(), df['Overall'].max()+2, 2)
plt.hist2d(x = df['Age'], y = df['Overall'], cmap="YlGnBu", bins=[bin_x, bin_y])
plt.colorbar()
plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')

plt.subplot(1,2,2)
plt.scatter(x = df['Age'], y = df['Overall'], alpha=0.25, marker='.')
plt.xlabel('Age (years)')
plt.ylabel('Overall Rating')
plt.show()


# ### Overall Rating vs Potential

# In[ ]:


plt.figure(figsize=[8,5])
sb.jointplot(x=df.Overall, y=df.Potential, kind='kde')
plt.show()


# ### Overall Rating vs Potential vs Age

# In[ ]:


plt.figure(figsize=[8,5])
plt.scatter(x=df.Overall, y=df.Potential, c=df.Age, alpha=0.25, cmap='rainbow' )
plt.colorbar().set_label('Age')
plt.xlabel('Overall Rating')
plt.ylabel('Potential')
plt.show()


# ### Segregating, Cleaning and Modifying required Data

# In[ ]:


df_opa = df[['ID', 'Name', 'Age', 'Overall', 'Potential', 'Value', 'International Reputation', 'Height', 'Weight', 'Position', 'Wage']]
df_opa.head()


# In[ ]:


df_opa.head()


# In[ ]:


def currencystrtoint(amount):
    new_amount = []
    for s in amount:
        list(s)
        abbr = s[-1]
        if abbr is 'M':
            s = s[1:-1]
            s = float(''.join(s))
            s *= 1000000
        elif abbr is 'K':
            s = s[1:-1]
            s = float(''.join(s))
            s *= 1000
        else:
            s = 0
        new_amount.append(s)
    return new_amount


# In[ ]:


df_opa['Value'] = currencystrtoint(list(df_opa['Value']))


# In[ ]:


df_opa['Wage'] = currencystrtoint(list(df_opa['Wage']))


# In[ ]:


def lengthstrtoint(length):
    new_length = []
    for h in length:
        if type(h) is str:
            list(h)
            h = (float(h[0])*12) + float(h[2:])
        new_length.append(h)
    return new_length


# In[ ]:


df_opa['Height'] = lengthstrtoint(list(df_opa['Height']))


# In[ ]:


mean_height = df_opa['Height'].mean()


# In[ ]:


df_opa.Height.loc[df_opa['Height'].isnull()] = mean_height


# In[ ]:


def weightstrtoint(weight):
    new_weight = []
    for w in weight:
        if type(w) is str:
            w = float(w[0:-3])
        new_weight.append(w)
    return new_weight


# In[ ]:


df_opa['Weight'] = weightstrtoint(list(df_opa['Weight']))


# In[ ]:


mean_weight = df_opa['Weight'].mean()
df_opa.Weight.loc[df_opa['Weight'].isnull()] = mean_weight


# In[ ]:


mean_internationa_rep = df_opa['International Reputation'].mean()
df_opa['International Reputation'].loc[df['International Reputation'].isnull()] = mean_internationa_rep


# In[ ]:


df_opa.describe()


# ### Dependencies of properties with each other

# In[ ]:


plt.figure(figsize=(20,15))
sb.pairplot(df_opa)


# ### Nature of dependency of Wage on different properties

# ### 1. Wage vs Overall (degree = 1)

# In[ ]:


sb.lmplot(data=df_opa, x='Overall', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 2. Wage vs Overall (degree = 2)

# In[ ]:


sb.lmplot(data=df_opa, x='Overall', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 3. Wage vs Overall (degree = 3)

# In[ ]:


sb.lmplot(data=df_opa, x='Overall', y='Wage', order=3, scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 4. Wage vs Age (degree = 1)

# In[ ]:


sb.lmplot(data=df_opa, x='Age', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 5. Wage vs Age (degree = 2)

# In[ ]:


sb.lmplot(data=df_opa, x='Age', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'})


# ### 6. Wage vs Value (degree = 1)

# In[ ]:


sb.lmplot(data=df_opa, x='Value', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 7. Wage vs Height

# In[ ]:


sb.lmplot(data=df_opa, x='Height', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# In[ ]:


sb.lmplot(data=df_opa, x='Height', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 8. Wage vs Weight

# In[ ]:


sb.lmplot(data=df_opa, x='Weight', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# In[ ]:


sb.lmplot(data=df_opa, x='Weight', y='Wage', order=2, scatter_kws={'alpha':0.3, 'color':'y'} )


# ### 9. Wage vs International Reputation

# In[ ]:


sb.lmplot(data=df_opa, x='International Reputation', y='Wage', scatter_kws={'alpha':0.3, 'color':'y'} )


# ## Adding dummy variables for Position

# In[ ]:


df_opa = pd.get_dummies(df_opa, columns=['Position'], drop_first=True)


# In[ ]:


df_opa.info()


# In[ ]:


df_opa.describe()


# ## Wage Prediction

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics


# In[ ]:


X = df_opa.drop(['ID', 'Name', 'Wage'], axis=1)
y = df_opa['Wage']


# In[ ]:


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


from sklearn.preprocessing import StandardScaler
stsc = StandardScaler()
Xtrain = stsc.fit_transform(Xtrain)
Xtest = stsc.fit_transform(Xtest)


# In[ ]:


def pred_wage(degree, Xtrain, Xtest, ytrain):
    if degree > 1:
        poly = PolynomialFeatures(degree = degree)
        Xtrain = poly.fit_transform(Xtrain)
        Xtest = poly.fit_transform(Xtest)
    lm = LinearRegression()
    lm.fit(Xtrain, ytrain)
    wages = lm.predict(Xtest)
    return wages


# ### Using Linear Regression

# In[ ]:


predicted_wages1 = pred_wage(1, Xtrain, Xtest, ytrain)


# In[ ]:


sb.regplot(ytest, predicted_wages1, scatter_kws={'alpha':0.3, 'color':'y'})
plt.xlabel('Actual Wage')
plt.ylabel('Predicted Wage')
plt.show()


# In[ ]:


print('Mean Absolute Error : ' + str(metrics.mean_absolute_error(ytest, predicted_wages1)))
print('Mean Squared Error : ' + str(metrics.mean_squared_error(ytest, predicted_wages1)))
print('Root Mean Squared Error : ' + str(np.sqrt(metrics.mean_squared_error(ytest, predicted_wages1))))


# ### Using Polynomial Regression of degree=2

# In[ ]:


predicted_wages2 = pred_wage(2, Xtrain, Xtest, ytrain)


# In[ ]:


sb.regplot(ytest, predicted_wages2, scatter_kws={'alpha':0.3, 'color':'y'})
plt.xlabel('Actual Wage')
plt.ylabel('Predicted Wage')
plt.show()


# In[ ]:


print('Mean Absolute Error : ' + str(metrics.mean_absolute_error(ytest, predicted_wages2)))
print('Mean Squared Error : ' + str(metrics.mean_squared_error(ytest, predicted_wages2)))
print('Root Mean Squared Error : ' + str(np.sqrt(metrics.mean_squared_error(ytest, predicted_wages2))))


# #### As we can see above all the error values as we moved to degree=2 regression, so we are going to use only degree=1 i.e. linear regression.

# ## Residual

# In[ ]:


sb.distplot(ytest-predicted_wages1, bins=200, hist_kws={'color':'r'}, kde_kws={'color':'y'})
plt.xlim(-50000, 50000)


# # Here We can conclude that the prediction is fairly good as residuals are mainly distributed near 0 but we can improve it by considering more factors from the original data set.

# In[ ]:





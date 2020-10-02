#!/usr/bin/env python
# coding: utf-8

# This notebook will be covering some data cleaning followed by managing missing or deliberately introduced NaN values where the values for some features were missing. Finally the PCA will be implemented to know how many variables can this data be downed to so that minimum loss of variance is observed.

# In[ ]:


### Primary necessary imports and reading the data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
df_raw = pd.read_csv("../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")


# In[ ]:


df_raw.info() ### get info about the data


# Seems like we are having some missing values which we will replace by Numpy's NaN for now.

# In[ ]:


df_nan = df_raw.replace('?', np.nan) 


# In[ ]:


df_nan.isnull().sum() ### check for null values in every column.


# In[ ]:


df = df_nan #temporary save


# As we can see from the info above, there are few columns which were of 'object' type. So we will convert them to numeric type. Converting doesn't hurt hurt us as long as the columns are integer type or float type.

# In[ ]:


df1 = df.apply(pd.to_numeric)


# Check for the info again. Now you will notice all the columns are either integer type or float type unlike previously where we also had object type variables.

# In[ ]:


df1.info()


# It's high time that we remove all the NaN values we introduced earlier and add something more robust in place of them. So for the continous variables we will filling in with the median of that particular column and for the discrete variables we will be using either 0 or 1.

# In[ ]:


####filling NaN values with median for continous variables and 0/1 for discrete variables.

df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
df['Smokes'] = df['Smokes'].fillna(1)
df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
df['IUD'] = df['IUD'].fillna(0)
df['IUD (years)'] = df['IUD (years)'].fillna(0)
df['STDs'] = df['STDs'].fillna(1)
df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())
df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())
df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())
df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())
df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())
df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())
df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())


# Now, we also have categorical type variables in our dataset. So we will be filling them with dummy values.

# In[ ]:


####filling NaN values with dummy values for categorical variables.

df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',
                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])


# Now check for any null or NaN values. As we can see there are no anomalies with the work done so far so we can move ahead.

# In[ ]:


df.isnull().sum()


# In[ ]:


df_final = df #temporary save


# In[ ]:


df.describe()


# Importing the PCA algorithm to reduce the number of features for the classification. You can play with the variance percentage to get a fair idea of where the Eigen values are going for different variance percentage. After a lot of computation I chose to stay with 0.8 because after that the Eigen values were going below 1 which in our case is useless, so drop the components after this threshold.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.drop('Biopsy', axis=1)
y = df["Biopsy"]

X = StandardScaler().fit_transform(X)  # Standardizing the values in X.

pca = PCA(0.80)  # Changes in variance percentage can be made here.
prin_comp = pca.fit_transform(X)
principalDf = pd.DataFrame(data = prin_comp)

print(principalDf)

print('\nEigenvalues \n%s' %pca.explained_variance_)
print('Eigenvectors \n%s' %pca.components_)


# This is where you can plot the PCA components to know where the Eigen values are being mapped to each Principal Component.

# In[ ]:


def scree_plot():
    from matplotlib.pyplot import figure, show
    from matplotlib.ticker import MaxNLocator

    ax = figure().gca()
    ax.plot(pca.explained_variance_)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.axhline(y=1, linewidth=1, color='r', alpha=0.5)
    plt.title('Scree Plot of PCA: Component Eigenvalues')
    show()

scree_plot()


# Now concatenate your Prinicipal Component dataframe with the Biopsy (the target variable) column of our original dataset.

# In[ ]:


finalDf = pd.concat([principalDf, df[["Biopsy"]]], axis = 1)
finalDf


# This is just a plot to show that it is really not worthy to reduce this dataset to 2 or perhaps 3 variables because there will be a lot of loss of variance and hence the essence of original data would be lost.

# In[ ]:


pca2 = PCA(n_components = 2)  # Changes can be made here.
prin_comp2 = pca2.fit_transform(X)
principalDf = pd.DataFrame(data = prin_comp2
             , columns = ['PC 1', 'PC 2'])

finalDf2 = pd.concat([principalDf2, df[['Biopsy']]], axis = 1)

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2['Biopsy'] == target
    ax.scatter(finalDf2.loc[indicesToKeep, 'PC 1']
               , finalDf2.loc[indicesToKeep, 'PC 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


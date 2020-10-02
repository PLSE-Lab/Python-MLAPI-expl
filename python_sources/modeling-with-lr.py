# %% [code]
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(color_codes=True)

# %% [code]

Pd_energy=pd.read_csv(filepath_or_buffer="../input/eergy-efficiency-dataset/ENB2012_data.csv",delimiter=",",header='infer',names=None)
Pd_energy.columns = ["relative_compactness", "surface_area", "wall_area", "roof_area", "overall_height", "orientation", "glazing_area", "glazing_area_distribution", "heating_load", "cooling_load"]

energy=np.genfromtxt("../input/eergy-efficiency-dataset/ENB2012_data.csv",delimiter=",",skip_header=1)
Pd_energy.head(5)

# %% [code]
len(Pd_energy.columns)

# %% [code]
print("Types of array elements : {}".format(energy.dtype))
print("Numbers of dimensions of array : {}".format(energy.ndim))
print("The size in each dimension : {}".format(energy.shape))
print("The total number of elements : {}".format(energy.size))

# %% [code]
for i in range(len(Pd_energy.columns)):
    print(Pd_energy.columns[i])
    print("mean is {}".format(energy[:,i].mean()))
    print("max  is {}".format(energy[:,i].max()))
    print("min  is {}".format(energy[:,i].min()))
    print("var  is {}".format(energy[:,i].var()))
    print("std  is {}".format(energy[:,i].std()))
    print("____________________________________________________")

# %% [code]
# check missing value
Pd_energy.isnull().sum()

# %% [code]
Pd_energy.info()

# %% [code]
Pd_energy.describe()

# %% [code]
# check normal dist for heating_load
fig, ax1 = plt.subplots()
sns.distplot(Pd_energy['heating_load'], ax=ax1, fit=stats.norm)
(mu, sigma) = stats.norm.fit(Pd_energy['heating_load'])
ax1.set(title='Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))

fig, ax2 = plt.subplots()
stats.probplot(Pd_energy['heating_load'], plot=plt)

# %% [code]
# check normal dist for cooling_load
fig, ax1 = plt.subplots()
sns.distplot(Pd_energy['cooling_load'], ax=ax1, fit=stats.norm)
(mu, sigma) = stats.norm.fit(Pd_energy['cooling_load'])
ax1.set(title='Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))

fig, ax2 = plt.subplots()
stats.probplot(Pd_energy['cooling_load'], plot=plt)

# %% [code]
# check Unique and freq for each col 
for i in Pd_energy.columns:
    print(Pd_energy[i].value_counts())

# %% [code]
# show the dist for all independent columns 
fig, axs = plt.subplots(2, 4, sharey=True,figsize=(8,10))
axs[0,0].hist(Pd_energy['relative_compactness'], bins =12)
axs[0,0].set_xlabel("relative_compactness")
axs[0,1].hist(Pd_energy['surface_area'])
axs[0,1].set_xlabel("surface_area")
axs[0,2].hist(Pd_energy['wall_area'])
axs[0,2].set_xlabel("wall_area")
axs[0,3].hist(Pd_energy['roof_area'])
axs[0,3].set_xlabel("roof_area")
axs[1,0].hist(Pd_energy['overall_height'])
axs[1,0].set_xlabel("overall_height")
axs[1,1].hist(Pd_energy['orientation'])
axs[1,1].set_xlabel("orientation")
axs[1,2].hist(Pd_energy['glazing_area'])
axs[1,2].set_xlabel("glazing_area")
axs[1,3].hist(Pd_energy['glazing_area_distribution'])
axs[1,3].set_xlabel("glazing_area_distribution")

# %% [code]
corr = Pd_energy.corr(method='pearson')
top_corr = corr.index[np.abs(corr["heating_load"]) > 0.55]
plt.figure(figsize=(14,6))
mask=np.zeros_like(Pd_energy[top_corr].corr())
mask[np.triu_indices_from(mask)]=True
print(top_corr)
sns.heatmap(Pd_energy[top_corr].corr(),cmap="RdYlGn_r",vmax=1.0,vmin=-1.0,mask=mask,annot=True,linewidths=2.5)

# %% [code]
pd.set_option('display.float_format',lambda x: '{:,.2f}'.format(x) if abs(x) < 10000 else '{:,.0f}'.format(x))
Pd_energy.corr()

# %% [code]

"""Visualize the relationship between the features and the responses."""

fig, axs = plt.subplots(2, 4, sharey=True,figsize=(8,10))
Pd_energy.plot(kind='scatter', x='relative_compactness', y='heating_load', ax=axs[0,0])
Pd_energy.plot(kind='scatter', x='surface_area', y='heating_load', ax=axs[0,1])
Pd_energy.plot(kind='scatter', x='wall_area', y='heating_load', ax=axs[0,2])
Pd_energy.plot(kind='scatter', x='roof_area', y='heating_load', ax=axs[0,3])
Pd_energy.plot(kind='scatter', x='overall_height', y='heating_load', ax=axs[1,0])
Pd_energy.plot(kind='scatter', x='orientation', y='heating_load', ax=axs[1,1])
Pd_energy.plot(kind='scatter', x='glazing_area', y='heating_load', ax=axs[1,2])
Pd_energy.plot(kind='scatter', x='glazing_area_distribution', y='heating_load', ax=axs[1,3])

# %% [code]
# create a fitted model with all three features
lm = smf.ols(formula='heating_load ~ relative_compactness + surface_area + wall_area + roof_area + overall_height + orientation + glazing_area + glazing_area_distribution', data=Pd_energy).fit()

# print the coefficients
lm.params

# %% [code]
lm.summary()

# %% [code]
#Normalize the inputs and set the output
from sklearn.preprocessing import Normalizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
nr = Normalizer(copy=False)
X= Pd_energy.drop(['heating_load','cooling_load'], axis=1)
y = Pd_energy[['heating_load','cooling_load']]

# %% [code]
# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 0)

# %% [code]
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# %% [code]
lr.coef_

# %% [code]
y_pred = lr.predict(X_test)

# %% [code]
df_1 = pd.DataFrame({'Actual': y_test["heating_load"], 'Predicted': y_pred[:,0]})
print(df.head(10))
print("\n______________________\n")
df_2 = pd.DataFrame({'Actual': y_test["cooling_load"], 'Predicted': y_pred[:,1]})
print(df.head(10))

# %% [code]
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr, X, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# %% [code]
from sklearn.metrics import mean_absolute_error ,mean_squared_error , r2_score
MSE=mean_squared_error(y_test,y_pred)
MAE=mean_absolute_error(y_test,y_pred) 
r2_value = r2_score(y_test, y_pred)                     

print("Intercept: \n", lr.intercept_)
print("Mean Square Error \n", MSE)
print("Mean Absolute Error \n", MAE)
print("R^2 Value: \n", r2_value)

# %% [code]

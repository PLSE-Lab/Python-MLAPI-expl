import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm 

# %% [code]
# i. Data Preprocessing
df = pd.read_csv('kc_house_data.csv')
df.fillna('',inplace=True)
df['sqft_living'].describe() 

# %% [code]
plt.figure(figsize = (9, 5)) 
df['sqft_living'].plot(kind ="hist") 

# %% [code]
# Correlation of each col with other cols
corrmat = df.corr() 
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 

# %% [code]
corrmat = df.corr() 
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0)  
cg 

# %% [code]
k = 7
cols = corrmat.nlargest(k, 'sqft_living')['sqft_living'].index 
cm = np.corrcoef(df[cols].values.T) 
f, ax = plt.subplots(figsize =(12, 10))  
sns.heatmap(cm, ax = ax, cmap ="YlGnBu", 
            linewidths = 0.1, yticklabels = cols.values,  
                              xticklabels = cols.values) 

# %% [code]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data
df.corr()
X = df[['sqft_above','grade','sqft_living15','bathrooms','price','bedrooms']]
y = df[['sqft_living']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)


# Train model
model = LinearRegression()
output_model = model.fit(X_train, y_train)
coef = model.coef_
print('Coefficients: ' + str(coef))
intercept = model.intercept_
print('Intercept: ' + str(intercept))

# Test model
y_pred = model.predict(X_test)

# Evaluation
R2 = r2_score(y_test, y_pred)
print('R2: ' + str(R2))
MSE = mean_squared_error(y_test, y_pred)
print('MSE: ' + str(MSE))

# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]


# %% [code]

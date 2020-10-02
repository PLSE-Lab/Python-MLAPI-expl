import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#Read the text file --> conver to lists --> convert to dataframe
results = []
with open('../input/wine.data.txt') as inputfile:
    for line in inputfile:
        results.append(line.strip().split(','))
        
##print(results)

labels = ['Cultivar_Type', 'Alcohol', 'Malic_acid', 'Ash','Alcalinity_of_Ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Pranthocyanins','Color_intensity','Hue','OD280/OD315','Proline']
data = pd.DataFrame.from_records(results, columns=labels)

print(data.head())

print(data.info()) #178 rows of data, no missing values, all are object type
data=pd.to_numeric(data, errors='ignore')
print(data.info())
"""
print(data.groupby('Cultivar_Type')['Cultivar_Type'].count())

#All columns are object type, convert them to float type/int type
for col in data:
    data[col] = data[col].astype('float64')
    
#Separate the data into three different dataframes, for each cultivar
data1=data[data['Cultivar_Type'] == 1]
data2=data[data['Cultivar_Type'] == 2]
data3=data[data['Cultivar_Type'] == 3]

print(data1.describe().loc[['mean','min','max']])
print(data2.describe().loc[['mean','min','max']])
print(data3.describe().loc[['mean','min','max']])
#Differences there within group in proline for e.g.


#Trying standardScalar on whole data
print(data.head())
Cultivar_Real=data['Cultivar_Type']
data.drop(['Cultivar_Type'],axis=1)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler=StandardScaler()
kmeans=KMeans(n_clusters=3)

from sklearn.pipeline import make_pipeline
pipeline=make_pipeline(scaler,kmeans)
pipeline.fit(data)
Cultivar_Predicted=pipeline.predict(data)

data_new=pd.DataFrame({'Predicted':Cultivar_Predicted,'Real':Cultivar_Real})
print(pd.crosstab(data_new['Predicted'],data_new['Real']))


#We are now going to check for multicollinearity between the independant variables
#VIF: Variance Inflation factors is the measure that helps identify the presence of multicollinearity
#The presence of multicollinearity acts as a noise in the data, so removing will most often give us better results with no significant change in R2
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
#gather features
features = "+".join(data.columns - ["Cultivar_Type"])

# get y and X dataframes based on this regression:
y, X = dmatrices('Cultivar_Type ~' + features, data, return_type='dataframe')

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

import statsmodels.formula.api as sm
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)

vif_cal(input_data=data, dependent_col="Cultivar_Type")
"""
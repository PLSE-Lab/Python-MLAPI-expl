#!/usr/bin/env python
# coding: utf-8

# # For previous work, based on which this notebook is made, go to [this link](https://www.kaggle.com/ashishsaxena2209/step-by-step-regression-backward-elimination).

# ## This notbook is for demonstration of application of polynomial features and some other concepts too. Mainly, it's for learning purpose.

# In[ ]:


# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import copy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import csv


# In[ ]:


odf = pd.read_csv('../input/diamonds/diamonds.csv')


# In[ ]:


odf.head()


# In[ ]:


# Dropping first column of dataframe
odf.drop(columns=odf.columns[0], axis = 1, inplace=True)
odf.head(10)


# In[ ]:


# Categorizing columns according to data types
categorical = odf.columns[[1,2,3]]
to_predict = 'price'
numeric = odf.columns[[0,4,5,6,7,8,9]]


# In[ ]:


odf.describe()


# In[ ]:


# Replacing zero values of x, y and z with NaNs. After then NaN will be dropped

odf[['x','y','z']] = odf[['x','y','z']].replace(0,np.NaN)

odf.dropna(inplace=True)


# In[ ]:


# Checking for missing values in dataset
for c in odf.columns:
    print('Total Missing values in \'{}\' are {}.'.format(c,odf[c].isna().sum()))


# In[ ]:


# For more details kindly refer to my previous notebook. Link provided at top.
odf.drop(odf[odf.y > 20].index, inplace=True) # Dropping outliers from 'y' column
odf.drop(odf[odf.z > 20].index, inplace=True) # Dropping outliers from 'z' column

plt.figure(figsize=(15,12))
for i in range(1,8):
    plt.subplot(3, 3, i)
    plt.scatter(odf['price'], odf[numeric[i-1]], s= 1)
    plt.xlabel(to_predict)
    plt.ylabel(numeric[i-1])


# ## From figures above it seems like few independent variables or features don't have linear relationship with the dependent variable.

# <a id='Orig_histogram'></a>
# # Original dataframe histogram

# In[ ]:


h = odf.hist(figsize=(15,10))


# In[ ]:


df_length = odf['carat'].count()
carat_count = odf['carat'][odf.carat > 3.0].count()
print('Only {}% data i.e. {} in number, is greater than 3.0 for carat column'.format(np.round((carat_count/df_length)*100, 2), carat_count))

table_count = odf['table'][odf.table > 70.0].count()
print('Only {}% data i.e. {} in number, is greater than 70.0 for table column'.format(np.round((table_count/df_length)*100, 2), table_count))

x_count = odf['x'][odf.x > 8.5].count()
print('Only {}% data i.e. {} in number, is greater than 8.5 for x column'.format(np.round((x_count/df_length)*100, 2), x_count))

y_count = odf['y'][odf.y > 8.5].count()
print('Only {}% data i.e. {} in number, is greater than 8.5 for y column'.format(np.round((y_count/df_length)*100, 2), y_count))

z_count = odf['z'][odf.z > 5.2].count()
print('Only {}% data i.e. {} in number, is greater than 5.0 for z column'.format(np.round((z_count/df_length)*100, 2), z_count))


# In[ ]:


# Dropping outliers based on histogram
odf.drop(odf[odf.carat > 3.0].index, inplace=True) # Dropping outliers from 'carat' column
odf.drop(odf[odf.table > 70.0].index, inplace=True) # Dropping outliers from 'table' column
odf.drop(odf[odf.x > 8.5].index, inplace=True) # Dropping outliers from 'x' column
odf.drop(odf[odf.y > 8.5].index, inplace=True) # Dropping outliers from 'y' column
odf.drop(odf[odf.z > 5.2].index, inplace=True) # Dropping outliers from 'z' column


# In[ ]:


h = odf.hist(figsize=(15,10))


# In[ ]:


# Heatmap before introducing dummy variables
p = sns.heatmap(odf.corr('spearman'), center=0, cmap = 'RdYlGn')


# ### It is evident from above figure that there is a strong correlation between various independent (as assumed) variables.

# In[ ]:


# Making dummy variables for categorical Columns
odfd = pd.get_dummies(data=odf, columns=categorical)


# In[ ]:


# Dropping Extra caegoricals
lst = ['color_D','cut_Fair','clarity_IF']
idx = []
for i in lst:
    # Removing D color from color column, Fair cut and IF clarity column
    idx.append(odfd.columns.get_loc(i))
    
odfd.drop(columns=odfd.columns[idx], axis = 1, inplace=True)


# In[ ]:


# Rearranging columns of dataframe

col = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_Good', 'cut_Ideal', 'cut_Premium', 'cut_Very Good', 'color_E', 
       'color_F', 'color_G', 'color_H', 'color_I', 'color_J', 'clarity_I1', 'clarity_SI1', 'clarity_SI2', 'clarity_VS1',
       'clarity_VS2', 'clarity_VVS1', 'clarity_VVS2', 'price']

odfd = odfd[col]


# In[ ]:


# Heatmap after introduction of dummy variables
plt.figure(figsize=(25,12))
p = sns.heatmap(data=odfd.corr(method='spearman'), annot=True, cmap='RdYlGn', center=0)


# **Again it can be seen from above figure that carat is highly correlated to x, y, z and viceversa.**

# # Function to Automating backward elimination process

# In[ ]:


# Automating backward elimination technique

def DoBackwardElimination(the_regressor, X, y, minP2eliminate):
    
    assert np.shape(X)[0] == np.shape(y)[0], 'Length of X and y do not match'
    assert minP2eliminate > 0, 'Minimum P value to eliminate cannot be zero or negative'
    
    original_list = list(range(0, np.shape(the_regressor.pvalues)[0]))
    
    max_p = 100        # Initializing with random value of maximum P value
    i = 0
    r2adjusted = []   # Will store R Square adjusted value for each loop
    r2 = []           # Will store R Square value  for each loop
    
    previous_R2adjusted = the_regressor.rsquared_adj
    
    while max_p >= minP2eliminate:
        
        p_values = list(the_regressor.pvalues)
        r2adjusted.append(the_regressor.rsquared_adj)
        r2.append(the_regressor.rsquared)
        
        max_p = max(p_values)
        max_p_idx = p_values.index(max_p)
        
        if max_p_idx == 0:
            
            temp_p = set(p_values)
            
            # removing the largest element from temp list
            temp_p.remove(max(temp_p))
            
            max_p = max(temp_p)
            max_p_idx = p_values.index(max_p)
            
#             print('Index value 0 found!! Next index value is {}'.format(max_p_idx))
            
            if max_p < minP2eliminate:
                
                print('Max P value found less than 0.1 with 0 index ...Loop Ends!!')
                
                break
                
        if max_p < minP2eliminate:
            
            print('Max P value found less than 0.1 without 0 index...Loop Ends!!')
            
            break
        
        val_at_idx = original_list[max_p_idx]
        
        idx_in_org_lst = original_list.index(val_at_idx)
        
        original_list.remove(val_at_idx)
        
        print('Popped column index out of original array is {} with P-Value {}'.format(val_at_idx, np.round(np.array(p_values)[max_p_idx], decimals= 4)))
        
        print('==================================================================================================')
        
        X_new = X[:, original_list]
        
        the_regressor = smf.OLS(endog = y, exog = X_new).fit()
        
        if previous_R2adjusted < the_regressor.rsquared_adj:
            classifier_with_maxR2adjusted = the_regressor
            final_list_of_index = copy.deepcopy(original_list)
            previous_R2adjusted = the_regressor.rsquared_adj
            
    return classifier_with_maxR2adjusted, r2, r2adjusted, final_list_of_index


# # Function to calculate and plot residuals

# In[ ]:


def resiplot(y_original, y_predicted, delete_outlier = False, max_outlier_val = None):
    
    residual = y_original - y_predicted
    residnew = list(residual.ravel())
    
    if delete_outlier == True:
        assert max_outlier_val != None, 'Please insert \'max_outlier_val\''
        count = 0
        while max(residnew) > abs(max_outlier_val):
            count = count + 1
            residnew.remove(max(residnew))
            
        while min(residnew)< -abs(max_outlier_val):
            count = count + 1
            residnew.remove(min(residnew))
        print('Residuals with unreal values are {} i.e. only {}% of total test data.'.format(count, np.round((count/len(residnew)*100), 2)))

    plt.scatter(x = range(0, len(residnew)), y = residnew, s = 2, c = 'R')
    plt.plot([0,len(residnew)], [0,0], '-k')
    if delete_outlier == True:
        plt.ylim(-abs(max_outlier_val), abs(max_outlier_val))
    elif abs(max(residnew)) > abs(min(residnew)):
        plt.ylim(-max(residnew), max(residnew))
    else:
        plt.ylim(min(residnew), abs(min(residnew)))
        
    plt.title('Mean residual is {}'.format(np.round(np.mean(residnew),2)))


# In[ ]:


# Preprocessing data

X = odfd.iloc[:,:-1].values          # Selecting all columns except last one that is 'price'.
y = odfd['price'].values

# Scaling data in default range
mscalar = MinMaxScaler()
X_minmax_scaled = mscalar.fit_transform(X)

# Adding constant values at start of array X
X_minmax_scaled = np.append(arr = np.ones((X_minmax_scaled.shape[0], 1)).astype(int), values=X_minmax_scaled, axis=1)


# In[ ]:


# This code will be used only if doing manual elimination [Part 1]
X_lst = list(range(0, X.shape[1]))
X_opt = X_minmax_scaled[:,X_lst]
flag = 2 # Done intentionally in order to stop popping 5th column from input array X


# In[ ]:


# This code will be used only if doing manual elimination [Part 2]
# Eliminating columns according to P values from summary to make new X_opt
idx_to_pop = 5
if flag == 1:
    X_lst.pop(element_to_pop)
    flag = 99
    
X_opt = X_minmax_scaled[:,X_lst]


# In[ ]:


# Splitting data for taining and testing
X_train, X_test, y_train, y_test = train_test_split(X_opt, y, test_size=0.20, random_state=22)


# # OLS Regressor without polynomial fitting of data

# In[ ]:


# Fitting the regressor to simple data without polynomial features. 
regressor_SLR_OLS = smf.OLS(endog = y_train, exog = X_train).fit()

# Looking at the summary of regressor
# print(regressor_SLR_OLS.summary())
print('R-Squared for this regressor is {}'.format(np.round(regressor_SLR_OLS.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(regressor_SLR_OLS.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(regressor_SLR_OLS.fvalue, 3)))

resiplot(y_original=y_test, y_predicted=regressor_SLR_OLS.predict(X_test), delete_outlier=True, max_outlier_val=10000)


# # Function to fit polynomial features

# In[ ]:


# Function to fit polynomial features

def fit_poly(columns, degree, df):
    
    # Column is a list of features to be used with polynomial
    # degree is polynomial degree
    # df is preprocessed and clean dataframe containing relevant features and data including feature to be predicted.
    
    poly_f = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)   # Bias will be included later
    X_with_poly_features = df[columns].values # It's a numpy array
    df_without_poly_features = df.drop(columns=columns, inplace=False) # A dataframe
    
    X_without_poly_features = df_without_poly_features.iloc[:,:-1].values
    y = df_without_poly_features['price'].values
    
    # Fitting polynomial features on selected columns
    X_with_poly_features = poly_f.fit_transform(X_with_poly_features) # fitting polynomial features

    X = np.append(arr=X_with_poly_features, values=X_without_poly_features, axis=1) # Both matrices are appended into one
    
    return X, y


# # Three different OLS Regressors, with Polynomial fitting, for comparison

# In[ ]:


# Starting with 'carat' and 'x', 'y', 'z'
col_for_polyfit1 = ['carat','x','y','z'] # from dataframe 'odfd'

# Now including 'depth' also to previous features
col_for_polyfit2 = ['carat','x','y','z','depth'] # from dataframe 'odfd'

# Now including 'table' also to previous features
col_for_polyfit3 = ['carat','x','y','z','depth','table'] # from dataframe 'odfd'

degree = 2 # starting with second degree

# Fitting poly features
X1, y = fit_poly(columns=col_for_polyfit1, degree=degree, df=odfd)
X2, y = fit_poly(columns=col_for_polyfit2, degree=degree, df=odfd)
X3, y = fit_poly(columns=col_for_polyfit3, degree=degree, df=odfd)

# Scaling data in default range
mscalar1 = MinMaxScaler()
X_minmax_scaled = mscalar.fit_transform(X1)
mscalar2 = MinMaxScaler()
X_minmax_scaled = mscalar.fit_transform(X2)
mscalar3 = MinMaxScaler()
X_minmax_scaled = mscalar.fit_transform(X3)

# Adding constant column to X's in order to be used with OLS regressor
X1 = np.append(arr = np.ones((X1.shape[0], 1)).astype(int), values=X1, axis=1)
X2 = np.append(arr = np.ones((X2.shape[0], 1)).astype(int), values=X2, axis=1)
X3 = np.append(arr = np.ones((X3.shape[0], 1)).astype(int), values=X3, axis=1)


# In[ ]:


# Splitting fitted X's and y's into testing and training sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.20, random_state=22)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.20, random_state=22)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.20, random_state=22)


# In[ ]:


# Fitting OLS regressor to training data
poly_regressor1 = smf.OLS(endog = y1_train, exog = X1_train).fit()
poly_regressor2 = smf.OLS(endog = y2_train, exog = X2_train).fit()
poly_regressor3 = smf.OLS(endog = y3_train, exog = X3_train).fit()


# # Summary of first OLS Regressor, with 2nd degree polyfit over 'carat', 'x', 'y' and 'z'

# In[ ]:


# Printing summary of poly_regressor1
# print(poly_regressor1.summary())
print('R-Squared for this regressor is {}'.format(np.round(poly_regressor1.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(poly_regressor1.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(poly_regressor1.fvalue, 4)))

y1_test_pred = poly_regressor1.predict(X1_test)
resiplot(y_predicted=y1_test_pred, y_original=y1_test,delete_outlier=True, max_outlier_val=10000)


# ### So, just by using polynomial feature with 2nd degree, R Square increased from 0.927 to 0.932. Backward Elimination yet to be used to see maximum R Square we can get. Let's see summary for other regressors also.

# # Summary of first OLS Regressor, with 2nd degree polyfit over 'carat', 'x', 'y' and 'z' also including 'depth'

# In[ ]:


# Printing summary of poly_regressor2
# print(poly_regressor2.summary())
print('R-Squared for this regressor is {}'.format(np.round(poly_regressor2.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(poly_regressor2.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(poly_regressor2.fvalue, 4)))

y2_test_pred = poly_regressor2.predict(X2_test)
resiplot(y_predicted=y2_test_pred, y_original=y2_test,delete_outlier=True, max_outlier_val=10000)


# ### Same R Square i.e. 0.932. F Value decreased which is not a good thing. No Backward Elimination used.

# # Summary of first OLS Regressor, with 2nd degree polyfit over 'carat', 'x', 'y' and 'z' also including 'depth' and 'table'

# In[ ]:


# Printing summary of poly_regressor3
# print(poly_regressor3.summary())
print('R-Squared for this regressor is {}'.format(np.round(poly_regressor3.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(poly_regressor3.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(poly_regressor3.fvalue, 4)))

y3_test_pred = poly_regressor3.predict(X3_test)
resiplot(y_predicted=y3_test_pred, y_original=y3_test,delete_outlier=True, max_outlier_val=10000)


# # Function for saving and loading classifiers using pickle

# In[ ]:


# Function of saving and loading classifier using pickle

import pickle 
  
def StoreData(regressor, filename = 'defaultfile.clf'): 
      
    # File is in Binary mode
    filepath = '../input/polynomial-regression-backward-elimination/' + filename
    
    if os.path.exists(filepath):
        print('File already exists!!')
    
    else:
        file = open(filename, 'ab')
        pickle.dump(regressor, file)                      
        file.close()
        print('File is saved by name {}'.format(filename))
    
def LoadData(filename_2bloaded = 'defaultfile.clf'): 
    
    filepath = '../input/polynomial-regression-backward-elimination/' + filename_2bloaded
    
    if os.path.exists(filepath):
        file = open(filepath, 'rb')
        regress = pickle.load(file)
        file.close()
    else:
        print('File does not exists!!')

    return regress


# ### Now let's change polynomial degree to 7 and see what happens or how much accuracy we get (If it runs!!). Polyfitting of data is done for columns 'carat', 'x', 'y', 'z', 'depth' and 'table'.

# In[ ]:


degree = 7 # starting with second degree

col_for_polyfit3 = ['carat','x','y','z','depth','table'] # from dataframe 'odfd'

# Fitting poly features
X4, y = fit_poly(columns=col_for_polyfit3, degree=degree, df=odfd)

# Scaling data in default range
mscalar_new = MinMaxScaler()
X4_minmax_scaled = mscalar_new.fit_transform(X4)


# Adding constant column to X's in order to be used with OLS regressor
X4_minmax_scaled = np.append(arr = np.ones((X4_minmax_scaled.shape[0], 1)).astype(int), values=X4_minmax_scaled, axis=1)

# Splitting fitted X's and y's into testing and training sets
X4_train, X4_test, y4_train, y4_test = train_test_split(X4_minmax_scaled, y, test_size=0.3, random_state=22)


# In[ ]:


import gc
gc.collect()


# In[ ]:


# Fitting OLS regressor to training data
# poly_regressor4 = smf.OLS(endog = y4_train, exog = X4_train).fit()

# Fitting of this classifier is time consuming and already done. This time loading saved classifier.
poly_regressor4 = LoadData(filename_2bloaded='Polyregressorwith7degree_v9.clf')


# # Summary of first OLS Regressor, with 7th degree polyfit over 'carat', 'x', 'y' and 'z' also including 'depth' and 'table'

# In[ ]:


# Printing summary of poly_regressor4
# print(poly_regressor4.summary())
print('R-Squared for this regressor is {}'.format(np.round(poly_regressor4.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(poly_regressor4.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(poly_regressor4.fvalue, 4)))

y4_test_pred = poly_regressor4.predict(X4_test)
resiplot(y_predicted=y4_test_pred, y_original=y4_test,delete_outlier=True, max_outlier_val= 20000)


# ### R square value increased from 0.933 to 0.941. Now, we need to check wether backward elimination do any good to model.

# ### Using pickle to save trained classifier. This will save out lots of time to retrain classifier in case of loss of variables due to system crash etc.

# In[ ]:


# Already stored
# StoreData(poly_regressor4, filename= 'Polyregressorwith7degree_v9.clf')


# # Applying Backward Elimination

# ### Backward Elimination is already done and classifier is saved so not going to do whole process again because it's a time taking thing. Just loading the classifier.

# In[ ]:


# classifier_with_maxR2adjusted, r2, r2adjusted, final_list_of_index = DoBackwardElimination(the_regressor=poly_regressor4, X= X4_train, 
#                                                                                             y= y4_train, minP2eliminate= 0.11)

# Backward elimination is a time consuming process thus we will not run it now. It is already done and needs to be loaded only.


# In[ ]:


# Storing optimized classifier to  be used further
# StoreData(filename='optimized_classifier_v9.clf', regressor=classifier_with_maxR2adjusted)


# In[ ]:


# Loading optimized classifier for further use
classifier_with_maxR2adjusted = LoadData(filename_2bloaded='optimized_classifier_v9.clf')


# In[ ]:


## Already done. Not required now

# Saving values of R Square, R Square Adjusted and final index list
# r2.insert(0, 'R2')
# r2adjusted.insert(0, 'R2Adjusted')

# import csv

# csvData = list(zip(r2,r2adjusted))
# with open('r2andr2adjusted_v9.csv', 'w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csvData)
    
# csvFile.close()

# iter_listofindex = []
# for i in final_list_of_index:
#     iter_listofindex.append([i])

# with open('final_index_list_v9.csv', 'w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(iter_listofindex)
    
# csvFile.close()


# In[ ]:


# Importing data into list from csv file

r2 = []
r2adjusted = []
final_list_of_index = [] # Remaining indexes after elimination
flag = 0
# os.listdir('../input/polynomial-regression-backward-elimination')
with open('../input/polynomial-regression-backward-elimination/r2andr2adjusted_v9.csv') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        if flag != 0:
            r2.append(float(row[0]))
            r2adjusted.append(float(row[1]))
        flag = 1
        
with open('../input/polynomial-regression-backward-elimination/final_index_list_v9.csv') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        final_list_of_index.append(int(row[0]))


# In[ ]:


# Checking values of R Square and R Square Adjusted
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0,len(r2adjusted), 1), r2adjusted, '-k')
plt.plot(np.arange(0,len(r2adjusted), 1), r2, '--r')

plt.xlim(0, len(r2)-1)
plt.legend([r'$R^2  Adjusted$', r'$R^2$'], shadow = True)
# p = plt.xticks(np.arange(0,len(r2), 1))

p = plt.text(50, 0.9412, r'Max value of $R^2$ is {}'.format(np.round(max(r2), 4)))
p = plt.text(50, 0.9398, r'Max value of Adjusted $R^2$ is {}'.format(np.round(max(r2adjusted), 4)))


# In[ ]:


# classifier_with_maxR2adjusted.summary()
print('R-Squared for this regressor is {}'.format(np.round(classifier_with_maxR2adjusted.rsquared, 3)))
print('R-Squared Adjusted for this regressor is {}'.format(np.round(classifier_with_maxR2adjusted.rsquared_adj, 3)))
print('F- Value for this regressor is {}'.format(np.round(classifier_with_maxR2adjusted.fvalue, 4)))

resiplot(y_original= y4_test, 
         y_predicted=classifier_with_maxR2adjusted.predict(X4_test[:,final_list_of_index]), delete_outlier=True,max_outlier_val= 10000)


# ### Though after using backward elimination technique, there is no apparent improvement in model or regressor, but one thing to remember is that size of input data reduced from 1733 columns to 1391 columns i.e. by 342 without reducing it's R Square value.

# In[ ]:


print('Total columns in original input for 7 degree polyfit are {}.'.format(np.shape(X4_test)[1]))
print('Total columns after optimization, in input for 7 degree polyfit are {}.'.format(len(final_list_of_index)))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Attrition Analysis
# 
# The dataset contains a total of 35 fields and 1470 observations. This analysis is based on the 'Attrition' field. This notebook attempts to identify some key features which can help explain 'Attrition' in the organisation. Gradient Boosting has been used to identify the important features to predict 'Attrition'.
# 
# ### Key Steps
# 
# The following key steps have been followed:
# 1. Exploratory Data Analysis
# 2. Feature Engineering
# 3. Model Fitting
# 4. Visualisation and Identification of Important Features
# 
# ## Step 1 - Exploratory Data Analysis (EDA)
# 
# This includes looking at the distribution of the target variable as well as identifying any missing values among the features.
# 
# First, all the required libraries are imported.

# In[ ]:


#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix)
#Suppressing warnings
import warnings
warnings.filterwarnings('ignore')


# As a next step, the Excel file has been imported using Pandas into the dataframe named 'df'. We can see that there are 26 numeric and 9 categorical fields.

# In[ ]:


#Importing  the Dataset
print('Importing the CSV file.')
df = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print('File imported successfully!')

#Datatypes in the dataset
print('Imported Dataframe Structure : \n', df.dtypes.value_counts())


# Using the head(), we can see a small snapshot of the data. The target field 'Attrition' is a categorical field with 'Yes' and 'No'.

# In[ ]:


df.head(3)


# We can clearly see that the 'Attrition' field is highly skewed with just over 200 'Yes' out of a total of 1470 observations. This skewness can result in a prediction which is highly geared towards predicting 'No'.

# In[ ]:


#Checking the number of 'Yes' and 'No' in 'Attrition'
ax = sns.catplot(x="Attrition", kind="count", palette="ch:.25", data=df);
ax.set(xlabel = 'Attrition', ylabel = 'Number of Employees')
plt.show()


# Next I check if there are any missing values in the dataframe. As can be seen from the output, there are no missing values in the dataset.

# In[ ]:


#Identifying columns with missing information
missing_col = df.columns[df.isnull().any()].values
print('The missing columns in the dataset are: ',missing_col)


# ## Step 2 - Feature Engineering
# 
# The numeric and categorical fields need to be treated separately and the target field needs to be separated from the training dataset. The following few steps separate the numeric and categorical fields and drops the target field 'Attrition' from the feature set.

# In[ ]:


#Extracting the Numeric and Categorical features
df_num = pd.DataFrame(data = df.select_dtypes(include = ['int64']))
df_cat = pd.DataFrame(data = df.select_dtypes(include = ['object']))
print("Shape of Numeric: ",df_num.shape)
print("Shape of Categorical: ",df_cat.shape)


# ### 2.1 Encoding Categorical Fields
# 
# The categorical fields have been encoded using the get_dummies() function of Pandas.

# In[ ]:


#Dropping 'Attrition' from df_cat before encoding
df_cat = df_cat.drop(['Attrition'], axis=1) 

#Encoding using Pandas' get_dummies
df_cat_encoded = pd.get_dummies(df_cat)
df_cat_encoded.head(5)


# ### 2.2 Scaling Numeric Fields
# 
# The numeric fields have been scaled next for best results. StandardScaler() has been used for the same. Post scaling of the numeric features, they are merged with the categorical ones.

# In[ ]:


#Using StandardScaler to scale the numeric features
standard_scaler = StandardScaler()
df_num_scaled = standard_scaler.fit_transform(df_num)
df_num_scaled = pd.DataFrame(data = df_num_scaled, columns = df_num.columns, index = df_num.index)
print("Shape of Numeric After Scaling: ",df_num_scaled.shape)
print("Shape of categorical after Encoding: ",df_cat_encoded.shape)


# In[ ]:


#Combining the Categorical and Numeric features
df_transformed_final = pd.concat([df_num_scaled,df_cat_encoded], axis = 1)
print("Shape of final dataframe: ",df_transformed_final.shape)


# In[ ]:


#Extracting the target variable - 'Attrition'
target = df['Attrition']

#Mapping 'Yes' to 1 and 'No' to 0
map = {'Yes':1, 'No':0}
target = target.apply(lambda x: map[x])

print("Shape of target: ",target.shape)

#Copying into commonly used fields for simplicity
X = df_transformed_final #Features
y = target #Target


# ### 2.3 Train and Test Split
# 
# The data is next split into training and test dataset using the train_test_split functionality of sklearn.

# In[ ]:


#Splitting into Train and Test dataset in 90-10 ratio
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8, random_state = 0, stratify = y)
print("Shape of X Train: ",X_train.shape)
print("Shape of X Test: ",X_test.shape)
print("Shape of y Train: ",y_train.shape)
print("Shape of y Test: ",y_test.shape)


# ## Step 3 - Model Fitting
# 
# Here, I am using the Gradient Boosting model for the decision trees to identify the importance of the features. 'Attrition' is the target variable.

# In[ ]:


#Using Gradient Boosting to predict 'Attrition' and create the Trees to identify important features
gbm = GradientBoostingClassifier(n_estimators = 200, max_features = 0.7, learning_rate = 0.3, max_depth = 5, random_state = 0, verbose = 0)
print('Training Gradient Boosting Model')

#Fitting Model
gbm.fit(X_train, y_train)
print('Model Fitting Completed')

#Predicting
print('Starting Predictions!')
y_pred = gbm.predict(X_test)
print('Prediction Completed!')


# In[ ]:


print('Accuracy of the model is:  ',accuracy_score(y_test, y_pred))


# We can obtain the confusion matrix to see how the model has performed. From the confusion matrix, we can see that the model has predicted 125 observations correctly and 22 observations incorrectly.

# In[ ]:


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('The confusion Matrix : \n',cm)


# ## Step 4 - Visualisation and Identification of Important Features
# 
# Here, I have used the 'feature_importances_' array of the Gradient Boosting Model to ascertain the most important features for the prediction of 'Attrition'.
# 
# From the plot below, we can clearly see that thet following features hold a lot of weightage:
# 1. Overtime
# 2. StockOptionLevel
# 3. JobSatisfaction
# 4. JobLevel
# 5. EnvironmentSatisfaction
# 6. TotalWorkingYears
# 
# We can next plot these individually alongside Attrition to better understand the importance of each.

# In[ ]:


# Scatter plot 
trace = go.Scatter(
    y = gbm.feature_importances_,
    x = df_transformed_final.columns.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 10,
        color = gbm.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = df_transformed_final.columns.values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Model Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter')


# ### 4.1 Overtime
# 
# From below, we can clearly see that Attrition is a bigger percentage of employees who did Overtime.

# In[ ]:


#Setting Seaborn font-size
sns.set(font_scale = 1)

#Attrition based on Overtime
ax = sns.catplot(x="OverTime", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Overtime', ylabel = 'Number of Employees', title = 'Overtime')
plt.show()


# ### 4.2 Stock Option Level
# 
# From below, we can clearly see that Attrition is higher for employees who dont have stock options. Employees with stock options are less likely to leave the organisation.

# In[ ]:


#Stock Option Level
ax = sns.catplot(x="StockOptionLevel", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Stock Option Level', ylabel = 'Number of Employees', title = 'Stock Option Level')
plt.show()


# ### 4.3 Job Satisfaction
# 
# While the number of Attrition is similar for all groupings, for Job Satisfaction of level 1, Attrition as percentage of employees in that group is higher than the others. So, employees experiencing lower satisfaction are more likely to leave.

# In[ ]:


#Job Satisfaction
ax = sns.catplot(x="JobSatisfaction", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Job Satisfaction', ylabel = 'Number of Employees', title = 'Job Satisfaction')
plt.show()


# ### 4.4 Job Level
# 
# We can see that the employees at lower levels are more likely to leave the organisation.

# In[ ]:


#JobLevel
ax = sns.catplot(x="JobLevel", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Job Level', ylabel = 'Number of Employees', title = 'Job Level')
plt.show()


# ### 4.5 Environment Satisfaction
# 
# While the number of Attrition is similar for all groupings, for Environment Satisfaction of level 1 and 2, Attrition as percentage of employees in those groups is higher than the others. So, employees experiencing lower satisfaction are more likely to leave.

# In[ ]:


#EnvironmentSatisfaction
ax = sns.catplot(x="EnvironmentSatisfaction", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Environment Satisfaction', ylabel = 'Number of Employees', title = 'Environment Satisfaction')
plt.show()


# ### 4.6 Years With Current Manager
# 
# Here we can see that employees with low number of years with current manager are more likely to leave. Attrition is high when the employee has spent not even 1 year with the current manager.

# In[ ]:


#YearsWithCurrManager
ax = sns.catplot(x="TotalWorkingYears", kind="count",hue="Attrition", palette="ch:.25", data=df);
ax.set(xlabel = 'Total Working Years', ylabel = 'Number of Employees', title = 'Total Working Years')
plt.show()


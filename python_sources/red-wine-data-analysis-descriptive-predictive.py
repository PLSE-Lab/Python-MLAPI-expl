#!/usr/bin/env python
# coding: utf-8

# ![Dataset](https://raw.githubusercontent.com/vikrantkakad/Red-Wine-Quality-Analysis/master/img/dataset.png)

# ### Setting up the development environment by importing required libraries and modules:
# - *Numpy:* It will provide the support for efficient numerical computation.  
# - *Pandas:* It is convenient library that supports dataframes. Working with pandas will bring ease in many crucial data operations.  
# - *Matplotlib:* It provides a MATLAB-like plotting framework.  
# - *Seaborn:* It is a visualization library based on matplotlib which provides a high-level interface for drawing attractive statistical graphics.  
# - *Bokeh:* It is a interactive visualization library that targets modern web browsers for presentation.  
# - *Statsmodel:* It provides functions and classes for statistical tests and models.  
# - *Sklearn:* It is python library for data mining, data analysis and machine learning.

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row
from bokeh.io import output_notebook
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import sklearn
import sklearn.metrics
from sklearn import ensemble
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')
output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the <span style="color:red"> Red Wine </span> dataset
# 
# - Lets read the red wine data set from the *'UCI Machine Learning Repository'*.  
# - Here, we can use the *read_csv()* from the *pandas* library to load data into dataframe from the remote url.

# In[21]:


url = "../input/winequality-red.csv"
wine = pd.read_csv(url)


# - The *head(..)* function of *pandas* helps in viewing the preview of the dataset for n-number of rows

# In[22]:


wine.head(n=5)


# ### Exploring the <span style="color:red">Red Wine</span> dataset:

# In[25]:


print("Shape of Red Wine dataset: {s}".format(s = wine.shape))
print("Column headers/names: {s}".format(s = list(wine)))


# - From above lines we can learn that there are total _1599 observations with 12 different feature variables/attributes_ present in the Red Wine dataset.

# In[ ]:


# Now, let's check the information about different variables/column from the dataset:
wine.info()


# - We can see that, all 12 columns are of numeric data types. Out of 12 variables, 11 are predictor variables and last one _'quality'_ is an response variable.

# In[ ]:


# Let's look at the summary of the dataset,
wine.describe()


# - The summary of Red Wine dataset looks perfect, there is no visible abnormality in data (invalid/negative values).
# - All the data seems to be in range (with different scales, which needs standardization).

# - Let's look for the missing values in red wine dataset:

# In[ ]:


wine.isnull().sum()


# - The red wine dataset doesn't have any missing values/rows/cells for any of the variables/feature.
# - It seems that data has been collected neatly or prior cleaning has been performed before publishing the dataset.

# - Let's rename the modify the dataset headers/column names by removing the _'blank spaces'_ from it.

# In[ ]:


wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
wine.head(n=5)


# #### Learning more about the target/response variable/feature:
# - Let's check how many unique values does the target feature _'quality'_ has?

# In[ ]:


wine['quality'].unique()


# - And how data is distributed among those values?

# In[ ]:


wine.quality.value_counts().sort_index()


# In[ ]:


sns.countplot(x='quality', data=wine)


# - The above distribution shows the range for response variable (_quality_) is between 3 to 8.

# - Let's create a new discreet, categorical response variable/feature ('_rating_') from existing '_quality_' variable.  
# _i.e._ bad: 1-4  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;average: 5-6  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;good: 7-10

# In[ ]:


conditions = [
    (wine['quality'] >= 7),
    (wine['quality'] <= 4)
]
rating = ['good', 'bad']
wine['rating'] = np.select(conditions, rating, default='average')
wine.rating.value_counts()


# In[ ]:


wine.groupby('rating').mean()


# #### Corelation between features/variables:
# - Let's check the corelation between the target variable and predictor variables,

# In[27]:


correlation = wine.corr()
plt.figure(figsize=(14, 8))
sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")


# In[ ]:


correlation['quality'].sort_values(ascending=False)


# - We can observe that, the *'alcohol, sulphates, citric_acid & fixed_acidity'* have maximum corelation with response variable '*quality*'. 
# - This means that, they need to be further analysed for detailed pattern and corelation exploration. Hence, we will use only these 4 variables in our future analysis. 

# #### Analysis of alcohol percentage with wine quality:

# In[ ]:


bx = sns.boxplot(x="quality", y='alcohol', data = wine)
bx.set(xlabel='Wine Quality', ylabel='Alcohol Percent', title='Alcohol percent in different wine quality types')


# #### Analysis of sulphates & wine ratings:

# In[ ]:


bx = sns.boxplot(x="rating", y='sulphates', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Sulphates', title='Sulphates in different types of Wine ratings')


# #### Analysis of Citric Acid & wine ratings:

# In[ ]:


bx = sns.violinplot(x="rating", y='citric_acid', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Citric Acid', title='Xitric_acid in different types of Wine ratings')


# #### Analysis of fixed acidity & wine ratings:

# In[ ]:


bx = sns.boxplot(x="rating", y='fixed_acidity', data = wine)
bx.set(xlabel='Wine Ratings', ylabel='Fixed Acidity', title='Fixed Acidity in different types of Wine ratings')


# #### Analysis of pH & wine ratings:

# In[ ]:


bx = sns.swarmplot(x="rating", y="pH", data = wine);
bx.set(xlabel='Wine Ratings', ylabel='pH', title='pH in different types of Wine ratings')


# ### Linear Regression:
# - Below graphs for different quality ratings shows a linear regression between residual_sugar & alcohol in red wine,

# In[ ]:


sns.lmplot(x = "alcohol", y = "residual_sugar", col = "rating", data = wine)


# - The linear regression plots above for different wine quality ratings (bad, average & good) shows the regression between alcohol and residual sugar content of the red wine.  
# - We can observe from the trendline that, for good and average wine types the residual sugar content remains almost constant irrespective of alcohol content value. Whereas for bad quality wine, the residual sugar content increases gradually with the increase in alcohol content.  
# - This analysis can help in manufacturing the good quality wine with continuous monitoring and contrilling the alcohol and residual sugar content of the red wine.

# In[ ]:


y,X = dmatrices('quality ~ alcohol', data=wine, return_type='dataframe')
print("X:", type(X))
print(X.columns)
model=smf.OLS(y, X)
result=model.fit()
result.summary()


# In[ ]:


model = smf.OLS.from_formula('quality ~ alcohol', data = wine)
results = model.fit()
print(results.params)


# - The above wine quality vs alcohol content regression model's result shows that, the minimum value for quality is 1.87 and there will be increment by single unit for wine quality for every change of 0.360842 alcohol units.

# ### Classification
# #### Classification using Statsmodel:
# - We will use statsmodel for this logistic regression analysis of predicting good wine quality (>4).
# - Let's create a new categorical variable/column (rate_code) with two possible values (good = 1 & bad = 0).

# In[ ]:


wine['rate_code'] = (wine['quality'] > 4).astype(np.float32)


# In[ ]:


y, X = dmatrices('rate_code ~ alcohol', data = wine)
sns.distplot(X[y[:,0] > 0, 1])
sns.distplot(X[y[:,0] == 0, 1])


# - The above plot shows the higher probability for red wine quality will be good if alcohol percentage is more than equal to 12, whereas the same probability reduces as alcohol percentage decreases.

# In[ ]:


model = smf.Logit(y, X)
result = model.fit()
result.summary2()


# In[ ]:


yhat = result.predict(X)
sns.distplot(yhat[y[:,0] > 0])
sns.distplot(yhat[y[:,0] == 0])


# In[ ]:


yhat = result.predict(X) > 0.955
print(sklearn.metrics.classification_report(y, yhat))


# - The above distribution plot displays the overlapped outcomes for the good and bad quality plots of the red wine.
# - We can observe that the precision for the good wine prediction is almost 96% accurate, where as for bad wine its only 4%, which is not good. But overall there is 92% average precision in wine quality rate prediction.

# #### Classification using Sklearn's LogisticRegression:

# In[ ]:


model = sklearn.linear_model.LogisticRegression()
y,X = dmatrices('rate_code ~ alcohol + sulphates + citric_acid + fixed_acidity', data = wine)
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))


# - The accuracy matrix for sklearn's linear regression model for red wine quality prediction shows the overall 92% precision which is similar to previous statsmodel's average precision.
# - Also the precision for good wine (1) prediction is almost 96%.
# - But the precision is almost 0% for the bad type of wine (0) with sklearn's linear regression model. Which is not a good sign for the analysis.

# #### Classification using Sklearn's RandomForestClassifier:

# In[ ]:


y, X = dmatrices('rate_code ~ alcohol', data = wine)
model = sklearn.ensemble.RandomForestClassifier()
model.fit(X, y)
yhat = model.predict(X)
print(sklearn.metrics.classification_report(y, yhat))


# - Here, with the accuracy matrix for sklearn's random forest classifier model for the prediction of red wine quality, we can observe that the values have been improved significantly.
# - The precision for the prediction of bad quality wine (0) is almost 100% where as the precision for prediction of good quality wine (1) is approximately 96%.
# - This sklearn's random forest classifier model also has the overall precision around 96%, which is far better than the previous two models (i.e. statsmodel and sklearn's linear regression model)

# ### Conclusion
# - We observed the key factors that determine and affects the quality of the red wine. Wine quality is ultimately a subjective measure. The ordered factor _'quality'_ was not very helpful and to overcome this, so we created another variable called _'rating'_.
# - To make predictions of wine quality and any other if required, we trained two models. As seen, the statsmodel and sklearn's Linear Regression model along with Random Forest Classifier. The Random Forest Classifier performed marginally better and we decided to stick with it if we had to make any more predictions.
# - The usage of this analysis will help to understand whether by modifying the variables, it is possible to increase the quality of the wine on the market. If you can control your variables, then you can predict the quality of your wine and obtain more profits.

# <img src="https://raw.githubusercontent.com/vikrantkakad/Red-Wine-Quality-Analysis/master/img/thank_you.jpg"/>

# In[ ]:





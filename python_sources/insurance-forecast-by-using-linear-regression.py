#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:


# Data Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
sns.set(style="darkgrid")

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# ### Check out the Data

# In[ ]:


# Read in the data
f = pd.read_csv('../input/insurance/insurance.csv')


# In[ ]:


# Basic info of Data set
f.info()


# In[ ]:


# Basic Statistics on numeric columns
f.describe()


# In[ ]:


# Read first 5 values using head()
f.head()


# # Exploratory data analysis
# 
# #### Let's create some simple plots to check out the data!

# In[ ]:


# Distribution of Charges for Female Sex
fig = go.Figure()

fig.add_trace(go.Histogram(x = f['charges'], y = f[f['sex'] == 'female']['charges'], marker_color='#5852a8',))

fig.update_layout(title='Distribution of Charges for Female Sex',bargap=0.05,
                  xaxis = dict(title = 'Charges'),
                  yaxis = dict(title = 'Number of Adult Females')
                 )
fig.show()


# In[ ]:


# Distribution of Charges for Male Sex
fig = go.Figure()

fig.add_trace(go.Histogram(x = f['charges'], y = f[f['sex'] == 'male']['charges'], marker_color='#5852a8',))

fig.update_layout(title='Distribution of Charges for Male Sex',bargap=0.05,
                  xaxis = dict(title = 'Charges'),
                  yaxis = dict(title = 'Number of Adult Males')
                 )
fig.show()


# In[ ]:


# Age Analysis

bins = [18, 36, 56]
names = ['Young Age Adults (18-35)', 'Middle Age Adults (36-55)', 'Elderly Age Adults (55+)']
d = dict(enumerate(names, 1))
f['Age_Category'] = np.vectorize(d.get)(np.digitize(f['age'], bins))

fig = go.Figure()

fig.add_trace(go.Histogram(x = f['Age_Category'], marker_color='#5852a8'))

fig.update_layout(title='Number of Adults Categorised by Age',
                  xaxis = dict(title = 'Age Category'),
                  yaxis = dict(title = 'Number of Adults'),
                  bargap=0.05
                 )
fig.show()


# In[ ]:


# BMI as per Age Category

fig = px.box(f,x = f['Age_Category'], y = f['bmi'],
             color= 'smoker', title="Box plot of BMI for Each Age Category (Categorised by Smoker & Non-smoker)"
             )
fig.show()


# In[ ]:


# Charges as per Age Category

fig = px.box(f,x = f['Age_Category'], y = f['charges'],
             color= 'smoker', title="Box plot of Charges for Each Age Category (Categorised by Smoker & Non-smoker)"
             )
fig.show()


# In[ ]:


# BMI as per Region
fig = px.box(f,x = f['region'], y = f['bmi'],
             color= 'smoker', title="Box plot of bmi for Each region (Categorised by Smoker & Non-smoker)"
             )
fig.show()


# In[ ]:


# transform Non-numerical labels (as long as they are hashable and comparable) to Numerical labels.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# region 
le.fit(f['region'].drop_duplicates()) 
f['region'] = le.transform(f['region'])

# smoker or not ( "1" for Yes)
le.fit(f['smoker'].drop_duplicates()) 
f['smoker'] = le.transform(f['smoker'])

# sex ("1" for Male)

le.fit(f['sex'].drop_duplicates()) 
f['sex'] = le.transform(f['sex'])

# Correlation Heatmap
df = f.drop(columns = 'Age_Category')
sns.heatmap(df.corr())


# # Linear Regression
# 
# #### Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the charges column.

# In[ ]:


# X and y arrays
from sklearn.preprocessing import PolynomialFeatures

X = f.drop(columns = ['charges','Age_Category'], axis = 1)
y = f['charges']

pf = PolynomialFeatures()
X_pf = pf.fit_transform(X)


# In[ ]:


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pf, y, test_size=0.25, random_state=0)


# In[ ]:


# Creating and Training the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
plr = lr.fit(X_train,y_train)


# In[ ]:


# Predictions from our Model
predictions = lr.predict(X_test)


# In[ ]:


# Let's grab predictions off our test set and see how well it did!

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test,
                y=predictions,
                marker_color='white',
                mode='markers'
                ))
fig.update_layout(title='Predicted Charges vs Actual Charges',
                  xaxis = dict(title = 'Actual Charges'),
                  yaxis = dict(title = 'Predicted Charges')
                  )
fig.show()


# In[ ]:


# Score
print(plr.score(X_test,y_test))


# ## Regression Evaluation Metrics
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[ ]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


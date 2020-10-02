#!/usr/bin/env python
# coding: utf-8

# # Estimate Data-Science Salary Around The World
# 
# Since **salary negotiations are hard** and good reasons for a raise are rarely based on large datasets of colleagues, I am going to fill this gap with this notebook.
# 
# Using the [**Stack Overflow Survey Dataset**](https://www.kaggle.com/stackoverflow/stack-overflow-2018-developer-survey/home) I am going to create a regression model for the salary of data scientists around the world.<br>
# Feel free to suggest further improvements or extensions.
# 
# Have a good day and good luck!
# 
# + [1. Import Libraries](#1)<br>
# + [2. Load Dataset](#2)<br>
# + [3. Slicing The Data For The Interesting Part](#3)<br>
# + [4. Cleaning The Data](#4)<br>
#  + [4.1. Salaries](#4.1)<br>
#  + [4.2. Countries](#4.2)<br>
#  + [4.3. Employment](#4.3)<br>
#  + [4.4. Company Size](#4.4)<br>
#  + [4.5. Years Coding](#4.5)<br>
#  + [4.6. Gender](#4.6)<br>
#  + [4.7. Age](#4.7)<br>
# + [5. Create Label, Train-, Valid- And Test-Set](#5)<br>
# + [6. Train And Optimize The Regressor](#6)<br>
# + [7. Explore Salary-Space](#7)<br>
#  + [7.1. Salary By Years Coding And Company Size](#7.1)<br>
#  + [7.2. Salary By Age And Company Size](#7.2)<br>
# + [8. Conclusion](#8)<br>
# 
# ***
# ## <a id=1>1. Import Libraries</a>

# In[ ]:


# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
import matplotlib.pyplot as plt

# To create interactive plots
import plotly.graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode
init_notebook_mode(True)

# To prepare training
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# To gbm light
from lightgbm import LGBMRegressor

# To optimize the hyperparameters of the model
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval


# ***
# ## <a id=2>2. Load Dataset</a>

# In[ ]:


# Load the dataset
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False)

print('Shape Dataset:\t{}'.format(df.shape))
df.sample(3)


# The dataset contains roughly **100.000 survey participants and about 130 questions.**<br>
# 
# ***
# ## <a id=3>3. Slicing The Data For The Interesting Part</a>
# 
# Since you have to answer the questions for yourself to predict your salary, I will restrict the questions to a small number of basic ones.

# In[ ]:


# Filter only good questions
df = df[['Country',
         'Employment',
         'CompanySize',
         'DevType',
         'YearsCoding',
         'ConvertedSalary',
         'Gender',
         'Age']]

print('Shape Dataset:\t{}'.format(df.shape))
df.sample(3)


# There are many **different jobs included in this survey.** Let us visualize them and filter for the interesting ones here on Kaggle.

# In[ ]:


# Split the jobs and count them
df_jobs = pd.DataFrame.from_records(df['DevType'].dropna().apply(lambda x: x.split(';')).values.tolist()).stack().reset_index(drop=True).value_counts()

# Create plot
df_jobs.plot(kind='barh', figsize=(10,7.5))
plt.title('Stack Overflow Survey Job-Count')
plt.xlabel('Job-Count')
plt.ylabel('Job')
plt.grid()
plt.show()


# Right in the middle you can see **two jobs (dark blue) working with data and machine learning.** I am filtering the DataFrame for both of these jobs.

# In[ ]:


# Filter for the right jobs
df = df[~df['DevType'].isna()]
df = df[df['DevType'].str.contains('Data ')].drop('DevType', axis=1)

print('Shape Dataset:\t{}'.format(df.shape))
df.sample(3)


# Right now the DataFrame contains only the important questions and the responses of datadriven jobs around the world.
# 
# In the next paragraph I will clean the data and preprocess it to fit into a regression model.
# 
# ***
# ## <a id=4>4. Cleaning The Data</a>
# ### <a id=4.1>4.1. Salaries</a>

# In[ ]:


# Empty values
print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))

# Create subplots
fig, axarr = plt.subplots(2, figsize=(10,7.5))

# Create histogram
df['ConvertedSalary'].hist(bins=100, ax=axarr[0])
axarr[0].set_title('Salary Histogram')
axarr[0].set_xlabel('Salary')
axarr[0].set_ylabel('Count')

# Create sorted plot
df['ConvertedSalary'].sort_values().reset_index(drop=True).plot(ax=axarr[1])
axarr[1].set_title('Ordered Salaries')
axarr[1].set_xlabel('Ordered Index')
axarr[1].set_ylabel('Salary')

plt.tight_layout()
plt.show()


# The very low salary cluster around zero and the odd peaks at high numbers seem suspicious.<br>
# I am gonna **remove salaries lower than 1000$ per year and higher ones than 490.000$.** The model will still cover most of the variability of the data but should be more robust.

# In[ ]:


# Remove suspiciously low and high salaries
df = df[(df['ConvertedSalary']>1000) & (df['ConvertedSalary']<490000)]
print('Shape Dataset:\t{}'.format(df.shape))


# ### <a id=4.2>4.2. Countries</a>

# In[ ]:


# Top n countries
n = 20

# Empty values
print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))

# Create plot
df_country = df['Country'].value_counts().head(n)
df_country.plot(kind='barh', figsize=(10,7.5))
plt.title('Count For The Top {} Countries'.format(n))
plt.xlabel('Count')
plt.ylabel('Country')
plt.grid()
plt.show()


# There seems to be an exponential decay in the number of participants in the different countries.<br>
# To have a reasonable amount of data left for each country **I will restrict my model to the five most frequent countries.**<br>
# If you like to expand the number of countries, remember the sparsity of the data increases with more countries and the inference will be less reliable.

# In[ ]:


# Filter for the most frequent countries
df = df[ df['Country'].isin( df_country[:5].index ) ]

print('Shape Dataset:\t{}'.format(df.shape))


# ### <a id=4.3>4.3. Employment</a>

# In[ ]:


# Empty values
print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))

# Create plot
df['Employment'].value_counts().plot(kind='barh', figsize=(10,5))
plt.title('Kind Of Employment')
plt.xlabel('Count')
plt.ylabel('Employment')
plt.show()


# Since the model will predict the salary of a person, I am **removing the retired and unemployed colleagues** from the dataset.<br>

# In[ ]:


# Impute and remove retired and unemployed colleagues
employment = ['Employed full-time', 
              'Employed part-time', 
              'Independent contractor, freelancer, or self-employed']
df = df[df['Employment'].fillna('Employed full-time').isin(employment)]

print('Shape Dataset:\t{}'.format(df.shape))


# ### <a id=4.4>4.4. Company Size</a>

# In[ ]:


# Ordered company sacle
company_size = ['Fewer than 10 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees', '1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees']

# Empty values
print('Empty Values:\t{}'.format(df['CompanySize'].isna().sum()))

# Create plot
df['CompanySize'].value_counts().reindex(company_size).plot(kind='barh', figsize=(10,7.5))
plt.title('Count Of Company Sizes')
plt.xlabel('Count')
plt.ylabel('Company Size')
plt.show()


# The **company size is an ordered category** and I will **transform it into a numerical column** instead of a dummy variable.<br>
# Imputation is not straight forward in this case since there is no obvious "right value". To not corrupt the ordered category and to keep predicting for new users simple, I will drop all ~300 empty values.

# In[ ]:


# Create mapping for company size
mapping_company_size = {key:i for i, key in enumerate(company_size)}

# Drop empty values
df = df.dropna(subset=['CompanySize'])

# Transform category to numerical column
df['CompanySize'] = df['CompanySize'].map(mapping_company_size)

print('Shape Dataset:\t{}'.format(df.shape))


# ### <a id=4.5>4.5. Years Coding</a>

# In[ ]:


# Ordered years coding sacle
years_coding = ['0-2 years', '3-5 years', '6-8 years', '9-11 years', '12-14 years', '15-17 years', '18-20 years', '21-23 years', '24-26 years', '27-29 years', '30 or more years']

# Empty values
print('Empty Values:\t{}'.format(df['YearsCoding'].isna().sum()))

# Create plot
df['YearsCoding'].value_counts().reindex(years_coding).plot(kind='barh', figsize=(10,7.5))
plt.title('Count Of Years Coding')
plt.xlabel('Count')
plt.ylabel('Years Coding')
plt.show()


# **Years coding is another ordered category** and I will transform it into a numerical column instead of a dummy variable again.<br>
# Luckily there are no empty values here.

# In[ ]:


# Create mapping for years coding
mapping_years_coding = {key:i for i, key in enumerate(years_coding)}

# Transform category to numerical column
df['YearsCoding'] = df['YearsCoding'].map(mapping_years_coding)


# ### <a id=4.6>4.6. Gender</a>

# In[ ]:


# Empty values
print('Empty Values:\t{}'.format(df['Gender'].isna().sum()))

# Create plot
df['Gender'].value_counts().plot(kind='barh', figsize=(10,7.5))
plt.title('Gender Count')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.show()


# Unfortunately the data is very sparse except for female and male participants. Therefore I will make use of **female and male only.**<br>
# Since there are roughly ten times more men than woman I will insert "Male" into the missing values.

# In[ ]:


# Impute and map gender
df['Gender'] = df['Gender'].fillna('Male')
df = df[df['Gender'].isin(['Male', 'Female'])]
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})

print('Shape Dataset:\t{}'.format(df.shape))


# ### <a id=4.7>4.7. Age</a>

# In[ ]:


# Ordered age sacle
age = ['Under 18 years old',
       '18 - 24 years old',
       '25 - 34 years old',
       '35 - 44 years old',
       '45 - 54 years old',
       '55 - 64 years old',
       '65 years or older']

# Empty values
print('Empty Values:\t{}'.format(df['Age'].isna().sum()))

# Create plot
df['Age'].value_counts().reindex(age).plot(kind='barh', figsize=(10,7.5))
plt.title('Age Count')
plt.xlabel('Count')
plt.ylabel('Age')
plt.show()


# The age is another **ordered category which will be converted to numeric values.** <br>
# I will insert the by far most frequent value of "25 - 34 years old" into the empty values.

# In[ ]:


# Create mapping for years coding
mapping_age = {key:i for i, key in enumerate(age)}

# Transform category to numerical column
df['Age'] = df['Age'].fillna('25 - 34 years old').map(mapping_age)

print('Shape Dataset:\t{}'.format(df.shape))


# ***
# ## <a id=5>5. Create Label, Train-, Valid- And Test-Set</a>
# 
# The dataset will be split into a **training, validation and testing set.**

# In[ ]:


# Create label
y = np.log(df['ConvertedSalary'].values)

# Create data
X = pd.get_dummies(df.drop('ConvertedSalary', axis=1)).values

# Create splitting of training and testing dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, test_size=0.25, random_state=1)

print('Training examples:\t\t{}\nExamples for optimization loss:\t{}\nFinal testing examples:\t\t{}'.format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))


# ***
# ## <a id=6>6. Train And Optimize The Regressor</a>
# 
# Now a basic **LGBMRegressor** will be trained and **automatically optimized.**

# In[ ]:


# Define function to minimize
def objective(args):
    # Create & fit model
    model = LGBMRegressor(**args)
    model.fit(X_train, y_train)
    
    # Predict testset
    y_pred = model.predict(X_valid)
    
    # Compute rmse loss
    loss = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_valid))
    return {'loss':loss, 'status':STATUS_OK, 'model':model}


# Setup search space
space = {'n_estimators': hp.choice('n_estimators', range(3000, 4500)),
         'max_depth': hp.choice('max_depth', range(25, 50)),
         'min_child_samples': hp.choice('min_child_samples', range(2, 10)),
         'reg_alpha': hp.uniform('reg_alpha', 0, 10),
         'reg_lambda': hp.uniform('reg_lambda', 1000, 3000)}


# Minmize function
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=1000, trials=trials, rstate=np.random.RandomState(1))

# Compute final loss
model = trials.best_trial['result']['model']
y_pred = model.predict(X_test)
loss = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_test))


# Print training results
for key, value in space_eval(space, best).items():
    print('{}:\t{}'.format(key, value))
print('\nTraining loss:\t{}'.format(trials.best_trial['result']['loss']))
print('\nFinal loss:\t{}'.format(loss))


# ***
# ## <a id=7>7. Explore Salary-Space</a>
# 
# I will create all possible answers and will predict the salary for them. Afterwards I am going to create an **interactive heatmap for your data exploration.** Of course you can switch between the countries by yourself.

# In[ ]:


# Create all possible answers
possibilities = []
for a in range(len(company_size)):
    for b in range(len(years_coding)):
        for c in range(len(['male', 'female'])):
            for d in range(len(age)):
                for e in range(len(['Canada', 'Germany', 'India', 'UK', 'US'])):
                    for f in range(len(['Fulltime', 'Parttime'])):
                        vector = np.zeros(11)
                        vector[0] = a
                        vector[1] = b
                        vector[2] = c
                        vector[3] = d
                        vector[4+e] = 1
                        vector[9+f] = 1
                        possibilities.append(vector)
possibilities = np.array(possibilities)

# Predict salaries for all answers
all_salaries = np.round(np.exp(model.predict(possibilities)), -2)

# Create data-structure for all salaries
df_plot = pd.DataFrame(possibilities, columns=['Size', 'Years', 'Gender', 'Age', 'Canada', 'Germany', 'India', 'UK', 'US', 'Full', 'Part'])
df_plot['Salary'] = all_salaries



# Create template for an interactive heatmap
def createHeatmap(x, y, x_axis, y_axis, x_label, y_label, filename):
    # Create hover texts & annotations
    def getAnotations(grid):
        hovertexts = []
        annotations = []
        for i, size in enumerate(y_axis):
            row = []
            for j, years in enumerate(x_axis):
                salary = grid[i, j]/1000
                row.append('Salary: {:.1f} k$/a<br>{}: {}<br>{}: {}<br>'.format(salary, y_label, size ,y_label, years))
                annotations.append(dict(x=years, y=size, text='{:.1f}'.format(salary), ax=0, ay=0, font=dict(color='#000000')))
            hovertexts.append(row)
        return hovertexts, annotations

    # Create traces
    data = []
    all_annotations = []
    # Iterate countries
    countries = ['US', 'UK', 'Germany', 'India', 'Canada']
    for i, country in enumerate(countries):
        # Get data
        grid = df_plot[df_plot[country]==1].pivot_table(index=y, columns=x, values='Salary', aggfunc='median').values
        # Get annotations
        hovertexts, annotations = getAnotations(grid)
        all_annotations.append(annotations)
        # Create trace
        trace = go.Heatmap(x = x_axis,
                           y = y_axis,
                           z = grid,
                           visible = True if i==0 else False,
                           text = hovertexts,
                           hoverinfo = 'text',
                           colorscale = 'Picnic',
                           colorbar = dict(title = 'Yearly<br>Salary',
                                           ticksuffix = '$'))
        data.append(trace)

    # Create buttons
    buttons = []
    # Iterate countries
    for i, country in enumerate(countries):
        label = country
        title = 'Median Salary Of A Data Scientist In {}'.format(country)
        visible = [False] * len(countries)
        visible[i] = True
        annotations = all_annotations[i]
        # Create button
        buttons.append(dict(label=label, method='update', args=[{'visible':visible},{'title':title, 'annotations':annotations}]))

    updatemenus = list([dict(type = 'dropdown',
                             active = 0,
                             buttons = buttons)])

    # Create layout
    layout = go.Layout(title = 'Median Salary Of A Data Scientist In {}'.format(countries[0]),
                       xaxis = dict(title = x_label,
                                    tickangle = -30),
                       yaxis = dict(title = y_label,
                                    tickangle = -30),
                       annotations = all_annotations[0],
                       updatemenus = updatemenus)

    # Create plot
    figure = go.Figure(data=data, layout=layout)
    plot(figure, filename=filename)
    iplot(figure)


# ### <a id=7.1>7.1. Salary By Years Coding And Company Size</a>

# In[ ]:


x = 'Years'
y = 'Size'
x_axis = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30 or more']
y_axis = ['<10',  '10-19', '20-99', '100-499', '500-999', '1,000-4,999', '5,000-9,999', '10,000<']
x_label = 'Years Coding'
y_label = 'Employees'

createHeatmap(x, y, x_axis, y_axis, x_label, y_label, 'Salary_Coding.html')


# ### <a id=7.2>7.2. Salary By Age And Company Size</a>

# In[ ]:


x = 'Age'
y = 'Size'
x_axis = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 years or older']
y_axis = ['<10',  '10-19', '20-99', '100-499', '500-999', '1,000-4,999', '5,000-9,999', '10,000<']
x_label = 'Age'
y_label = 'Employees'

createHeatmap(x, y, x_axis, y_axis, x_label, y_label, 'Salary_Age.html')


# ***
# ## <a id=8>8. Conclusion</a>
# 
# **Interpretation of the data for your specific case is up to you.**<br>
# Remember the predictions are based upon a survey and could be biased. Furthermore the model does not use all the possible features and the final plots show only the median for all results. For this reason there are **plenty of uncertainties in this notebook!**
# 
# I would be glad if you have some further suggestions for me.
# 
# **Have a good day!**

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import RepeatedKFold

np.random.seed(1)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# # Measuring the Applicants Gap in 2017
# 
# Let's do something really simple, but in a really effective manner. That thing is: measuring the applicants gaps in 2017.
# 
# ![image of actual amount of applicants vs. expected amont of applicants][1]
# 
# Each year, 8th graders from all over New York City take a test called SHSAT to see who will get access to the famous Specialized High Schools (SPHS). These Specialized High Schools are schools that receive great attention and provide a estimulating environment for students who have the desire to learn.
# 
# Problem is, students that receive offers from SPHS are usually from a very specific demographic. Usually white or asian guys, mostly from the upper class, and from very small number of schools. This demographic characteristics are tightly linked to academic performance<sup><a href="https://steinhardt.nyu.edu/scmsAdmin/media/users/sg158/PDFs/Pathways_to_elite_education/WorkingPaper_PathwaystoAnEliteEducation.pdf">1</a></sup>, showing deeply ingrained questions that cannot be solved in a simple manner.
# 
# What we can do, however, is target certain schools. PASSNYC is associated with a lot of organizations that provides services such as:
# 
# - Test preparation
# - Afterschool activities
# - Mentoring
# - Community and student groups
# - Etc
# 
# But, what schools to target? Simple, we choose those with the biggest gap between the actual number of applicants and the number of applicants that would be expected given the school characteristics.
# 
# ... 
# 
# Okay, it's not as simple as this. But, what we do here can provide invaluable information in the process of choosing which schools to intervene. This kernel is aimed at providing a simple yet effective model with which the gap can be estimated.
# 
# [1]: https://i.imgur.com/W4gYk7M.png

# # Little glimpse at the data
# 
# The crucial information for us is the amount of SHSAT applicants from each school. It can be found [here][1], and only includes students from 2017.
# 
# Then, we assembled lots of data related to the students and the schools they attend. Information includes:
# 
# - Percentage of each ethnicity in each school
# - Percentage of students with disabilities in each school
# - The Economic Need Index of each school (indicates the poverty of the students)
# - The distribution of grades for each school at the NYS Common Core tests<sup>1</sup>
# - Some other things
# 
# Below, you can see what I'm talking about more clearly.
# 
# <sub>1: The grades a student can get are 1, 2, 3 or 4. We have the percentage of students in each grade and also the mean scale score of each school. The mean scale score of each school has been standardized, for an easier interpretation.</sub>
# 
# [1]: https://data.cityofnewyork.us/Education/2017-2018-SHSAT-Admissions-Test-Offers-By-Sending-/vsgi-eeb5/

# In[ ]:


# import data
df = pd.read_pickle('../input/nycschools2017/schools2017.pkl')
print(df.shape[0], "schools")

# drop schools with missing test data
df = df[df.loc[:, 'Mean Scale Score - ELA':'% Level 4 - Math'].notnull().all(axis=1)]
print(df.shape[0], "schools after dropping missing test data")

# drop schools with missing attendance data
df = df[df['Percent of Students Chronically Absent'].notnull()]
print(df.shape[0], "schools after dropping missing attendance data")

# schools with 0-5 SHSAT testers have this value set to NaN
applicantsok = df['# SHSAT Testers'].notnull()

# show head of data
f2_columns = ['Latitude', 'Longitude', 'Economic Need Index',
              'Mean Scale Score - ELA', 'Mean Scale Score - Math']
pct_columns = [c for c in df.columns if c.startswith('Percent')]
pct_columns += [c for c in df.columns if c.startswith('%')]
df.head().style.     format('{:.2f}', subset=f2_columns).     format('{:.1%}', subset=pct_columns)


# # Principal Component Analysis
# 
# We expect our data to have a lot of collinearity. That is, we expect variables to be very related to one another. This might cause problems when fitting a model.
# 
# To alleviate this problem, we use a technique called Principal Component Analysis (abbreviated PCA). This technique reduces the amount of features to an arbitrary number (that we can specify). The features generated are those that can best explain the original variables, making it a really nice approach.

# # Cross-validation
# 
# To choose the best amount of features, we will use cross-validation. It is a technique for splitting the dataset into training and test sets multiple times, making use of the data in an efficient way.
# 
# The cross-validation method used is a [repeated k-fold][1]. Explaining it is out of the scope of this kernel, but, let's just say it is one of the most recommended methods. It requires a lot of iterations, but, since our dataset is not big (comparing to today's "big data"), this is no problem.
# 
# We used the parameters `n_splits=10` and `n_repeats=20`.
# 
# [1]: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold

# In[ ]:


# data
in_columns = [
    'Charter School?',
    'Percent Asian',
    'Percent Black',
    'Percent Hispanic',
    'Percent Other',
    'Percent English Language Learners',
    'Percent Students with Disabilities',
    'Economic Need Index',
    'Percent of Students Chronically Absent',
    
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math', 
]
inputs = df[applicantsok][in_columns]
outputs = logit(df[applicantsok]['% SHSAT Testers'])  # the logit will be explained later


# cross-validation
cv_results = []
n_splits = 10
n_repeats = 20
for n_components in range(1, inputs.shape[1] + 1):
    mae_scores = []
    mse_scores = []
    
    x = PCA(n_components).fit_transform(inputs)
    x = pd.DataFrame(x, index=inputs.index, columns=["PC{}".format(i) for i in range(1, n_components + 1)])
    x['Constant'] = 1
    y = outputs.copy()
    

    cv = RepeatedKFold(n_splits, n_repeats, random_state=1)    
    for train, test in cv.split(x):
        x_train = x.iloc[train]
        x_test = x.iloc[test]
        y_train = y.iloc[train]
        y_test = y.iloc[test]
        
        model = sm.RLM(y_train, x_train, M=sm.robust.norms.HuberT())
        results = model.fit()
        predictions = model.predict(results.params, exog=x_test)
        mae = median_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae_scores.append(mae)
        mse_scores.append(mse)
        
    mae_scores = np.array(mae_scores).reshape(n_repeats, n_splits).mean(axis=1)  # mean of each repeat
    mse_scores = np.array(mse_scores).reshape(n_repeats, n_splits).mean(axis=1)  # mean of each repeat
        
    mae_mean = np.mean(mae_scores)
    mae_std = np.std(mae_scores)
    mse_mean = np.mean(mse_scores)
    mse_std = np.std(mse_scores)
    
    cv_result = (n_components, mae_mean, mse_mean, mae_std, mse_std)
    cv_results.append(cv_result)
    
df_columns = ['n_components', 'mae__mean', 'mse__mean', 'mae__std', 'mse__std']
cv_results_df = pd.DataFrame(cv_results, columns=df_columns)
cv_results_df


# In[ ]:


# visualize results

cvdf = cv_results_df  # code sugar

plt.figure()
plt.errorbar(cvdf.n_components, cvdf.mae__mean, cvdf.mae__std, marker='o', label='Median Absolute Error')
plt.legend()

plt.figure()
plt.errorbar(cvdf.n_components, cvdf.mse__mean, cvdf.mse__std, marker='o', label='Mean Squared Error')
plt.legend();


# To choose the best number of features, we use both the Median Absolute Error and the Mean Squared Error. The first metric indicates how well the model is fitting overall, ignoring the weight of outliers<sup>1</sup>. The second metric, in contrast, is more sensitive to outliers, giving bigger importance to bigger errors.
# 
# Based on the plots, we can see that 8 principal components is a good choice. It has the lowest Median Absolute Error and still keeps the Mean Squared Error at control.
# 
# This is gonna be the value used.
# 
# <sub>1: A better explanation of how we should treat outliers is gonna be made in the modeling section</sub>

# # Fitting the model
# 
# We will use a very simple model. There are two gists, though:
# 
# 1. We are using it to *measure the application gap* in each school
# 
#    This may seem not relevant to the model choice, but it is very, very important.
# 
#    Say, of these two lines, which one do you think works best when predicting the gap between what is expected and what really occured?
# 
#    ![](https://i.imgur.com/DW4Rhcq.png)
#    ![](https://i.imgur.com/MfDYwnP.png)
# 
#   I'd say the second one, as it gives a nicer representation to points that are close to expected, and, predicts a big gap for points that are surely off.
# 
#   The first line was generated using a standard regression and the second one was generated using a robust regression (we say robust because it is *robust to outliers*). The model that I'm gonna use is a robust one.
# 
# 2. The outcomes we are trying to predict are *probabilities*
# 
#    Okay, they are actually the percentage of applicants at each school. But, can't we assume that this is the probability of each student at a certain school applying for the SHSAT? Although being a simplification, this is what we are gonna work upon. Being a model to predict probabilities, logistic regression is usually a better choice than linear regression. So, we use logistic regression<sup>1</sup>.
#    
# ---
# 
# In the end, the model we are gonna is a *robust logistic regression*<sup>2</sup>. In the next cells I will fit it, predict with it and display some results.
# 
# <sub>
#     1: I know, my logic is a little flawed. I will try improving on this area later.<br>
#     2: Actually, a robust linear regression with logits as outputs (transforming the model into a logistic regression)
# </sub>

# In[ ]:


base_df = df[[  # explanatory variables
    'Charter School?',
    'Percent Asian',
    'Percent Black',
    'Percent Hispanic',
    'Percent Other',
    'Percent English Language Learners',
    'Percent Students with Disabilities',
    'Economic Need Index',
    'Percent of Students Chronically Absent',
    
    'Mean Scale Score - ELA',
    '% Level 2 - ELA',
    '% Level 3 - ELA',
    '% Level 4 - ELA',
    'Mean Scale Score - Math',
    '% Level 2 - Math',
    '% Level 3 - Math',
    '% Level 4 - Math',
]]

# transform the variables (apply the PCA)
n_components = 8
pca = PCA(n_components)
transformed = pca.fit_transform(base_df)
transformed = pd.DataFrame(transformed, index=base_df.index, columns=["PC{}".format(i+1) for i in range(n_components)])

# add a constant column (needed for our model with statsmodels)
inputs = transformed
inputs.insert(0, 'Constant', 1.0)
inputs.head()


# In[ ]:


# prepare inputs and outputs
inputs_fit = inputs[applicantsok]
outputs_fit = logit(df['% SHSAT Testers'][applicantsok])
inputs_predict = inputs

# fit the model
model = sm.RLM(outputs_fit, inputs_fit, M=sm.robust.norms.HuberT())
results = model.fit()

# make predictions
predictions = model.predict(results.params, exog=inputs_predict)
predictions = pd.Series(predictions, index=inputs_predict.index)
predictions = expit(predictions)  # expit is the inverse of the logit
predictions.name = 'Expected % of SHSAT Testers'


# In[ ]:


results.summary()


# In[ ]:


_predictions = logit(predictions[applicantsok])  # values are in logit units
_actual = logit(df['% SHSAT Testers'][applicantsok])  # values are in logit units

xs = _predictions
ys = _actual - _predictions  # residual

plt.figure(figsize=(12, 8))
plt.plot(xs, ys, '.')
plt.axhline(0.0, linestyle='--', color='gray')
plt.xlim(-2.5, 2.5)
plt.ylim(-3.5, 3.5)
plt.title("Residual Plot (logit units)")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals");


# In[ ]:


mae = median_absolute_error(_actual, _predictions)
mse = mean_squared_error(_actual, _predictions)

print("Median Absolute Error:", mae)
print("Mean Squared Error:", mse)


# The low `P>|z|` values indicates that the model is really good, the residual plot indicates a healthy fit and the model scores are what we expected, given the cross-validation results.
# 
# Below I make a plot for the less statistic-oriented folks. It compares the percentage of students that were expected to take SHSAT to the percentage of students that actually took SHSAT<sup>1</sup>.
# 
# <sub>1: Schools from 0 to 5 test takers were not included.</sub>

# In[ ]:


xs = predictions[applicantsok]
ys = df['% SHSAT Testers'][applicantsok]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(xs, ys, s=5)
ax.plot([0, 1], [0, 1], '--', c='gray')
ax.xaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax.yaxis.set_major_formatter(plt.FuncFormatter("{:.0%}".format))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Regression Results")
ax.set_xlabel("Estimated Percentage of SHSAT Applicants")
ax.set_ylabel("Actual Percentage of SHSAT Applicants");


# Personally, I believe this is a good fit and everyone on Kaggle should use it. Kidding! I mean, it might probably be of good use to PASSNYC. :)
# 
# Below I will export the predicted probabilities to a nicer format.

# In[ ]:


df_export = predictions.to_frame()
df_export.to_csv("expected_testers.csv")
df_export.head()


# And it's done.

# # Thanks/Finishing remarks
# 
# I'd like to thank everyone that took part in this challenge, know that every little thing that you did gave ideas and inspiration that kept me going. I'd like to thank Chris Crawford, that put this thing to work, and all the people at PASSNYC that are working together towards a great objective (especially those who took part in this competition). Also, thanks to my friend Zin, who helped me a lot, and to my parents, who had the great ear to listen to everything I say.
# 
# I know this is past the deadline. If the modifications made are not considered valid, you can use the kernel versions as of August 7th (I was so rushed up).
# 
# Here are my other kernels for this competition:
# 
# - [Measuring the Applicants Gap (dropped schools)][1]: Same steps as this kernel, but for 14 schools which have some values missing
# - [Meaning/Usage of Applicants Gap][2]: A down to earth explanation of what the gap actually means
# - [Gaps along the years][3]: Predicting the gaps for 2015 and 2016 and measuring their distance
# - [Objective/Strategy formulation][4]: A view of the PASSNYC challenge as a whole
# 
# And the whole project, on github:
# 
# - https://github.com/araraonline/kag-passnyc
# 
# Cya!
# 
# [1]: https://www.kaggle.com/araraonline/measuring-the-applicants-gap-dropped-schools
# [2]: https://www.kaggle.com/araraonline/meaning-usage-of-applicants-gap
# [3]: https://www.kaggle.com/araraonline/gaps-along-the-years
# [4]: https://www.kaggle.com/araraonline/objective-strategy-formulation

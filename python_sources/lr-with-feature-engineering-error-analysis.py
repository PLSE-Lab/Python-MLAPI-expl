#!/usr/bin/env python
# coding: utf-8

# #  Linear regression with feature engineering, error analysis and outlier pruning

# In[ ]:


import pandas as pd
from plotly.offline import init_notebook_mode, iplot_mpl, download_plotlyjs, init_notebook_mode, plot, iplot
import plotly_express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression as lm
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as sm
from statsmodels.compat import lzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import statsmodels.api as sm
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import plotly.figure_factory as ff
# Import plotting modules
import matplotlib.pyplot as plt
init_notebook_mode(connected=True)
import pandas_profiling


# FEATURES :
# 
# - **sex:** insurance contractor gender, female, male
# 
# - **bmi:** body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# - **children:** Number of children covered by health insurance / Number of dependents
# 
# - **smoker:** Smoking
# 
# - **region:** the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# 
# TARGET:
# 
# - **charges:** Individual medical costs billed by health insurance

# In[ ]:


df = pd.read_csv('../input/insurance.csv')


# In[ ]:


pandas_profiling.ProfileReport(df)


# # 1. Dataset EDA

# ### 1.1 DataSet Descriptive statistics

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


import plotly.graph_objects as go
labels1 = ['Male', 'Female']
labels2= ['Smoker','Non Smoker']
labels3= ['southeast','northwest','southwest','northeast']
labels4= ['0 child','1  child','2 children','3 children','4 children','5 children']

# Create subplots, using 'domain' type for pie charts
specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
fig = make_subplots(rows=2, cols=2, specs=specs)

# Define pie charts
fig.add_trace(go.Pie(labels=labels1, values=[676,662], name='Sex'), 1, 1)
fig.add_trace(go.Pie(labels=labels2, values=[274,1064], name='Smoker'), 1, 2)
fig.add_trace(go.Pie(labels=labels3, values=[364,325,325,324], name='Region'), 2, 1)
fig.add_trace(go.Pie(labels=labels4, values=[574,324,240,157,25,18], name='Children'), 2, 2)

# Tune layout and hover info
fig.update_traces(hoverinfo='label+percent+name', textinfo='value', textfont_size=20,
                  marker=dict(line=dict(color='#000000', width=2)))

fig.update(layout_title_text='Categorical variables counts 1338 n-samples distribution')
fig = go.Figure(fig)
fig.show()


# The DataSet has a very symmetric distribution regarding sex and region. However, smokers make up only 20.5% of the sample data.

# ### 1.2 DataSet Correlation Matrix 
# We can explore the correlation matrix of our features.

# In[ ]:


df1=df.copy()
le = LabelEncoder()
le.fit(df1.sex.drop_duplicates()) 
df1.sex = le.transform(df1.sex)
# smoker or not
le.fit(df1.smoker.drop_duplicates()) 
df1.smoker = le.transform(df1.smoker)
#region
le.fit(df1.region.drop_duplicates()) 
df1.region = le.transform(df1.region)


corr = df1.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(12, 10))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, linewidths=.5, cmap='Reds', annot=True, fmt=".2f")
#Apply xticks
#show plot
plt.show()


# - We see that being a smoker seems has a strong positive correlation with higher medical costs.
# - Age and bmi also have a positive correlation, altough less strong.

# ### 1.3 Continuous variables distributions

# In[ ]:


px.histogram(df, x="charges",nbins=int(np.sqrt(len(df.charges))),
             marginal="box",title='Histogram & Box-Plot: Charges',
             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black"))


# - The distribution of the medical costs is very asymetric, with a very long right tail. 
# - 75% of the values are under 16.650$ and many outliers fall toward the right extreme. 
# - A logarytmic transformation could make the variable less skewed. 

# In[ ]:


px.histogram(df, x="bmi",nbins=int(np.sqrt(len(df.charges))),
             marginal="box",title='Histogram & Box-Plot: BMI',
             template='ggplot2').update_traces(dict(marker_line_width=1, marker_line_color="black", 
                                                    marker_color='darkcyan'))


# - We see that the BMI distribution resembles a normal distribution.

# In[ ]:


px.histogram(df, x="age",nbins=int(np.sqrt(len(df.charges))),
             marginal="box",title='Histogram & Box-Plot: Age',
             template='ggplot2').update_traces(dict(marker_line_width=1,marker_line_color="black", 
                                                    marker_color='violet'))


# - 18/19 year old are the most represented age group with 137 observations.
# - Only 22 observations fall to the 64-65 age group. This would be a problem for generalization.

# ### 1.3 Dependencies and interactions between features
# 
# Interactions occur when variables act together to impact the output of the process. We can use interaction plots are used to understand the behavior of how one variable (discrete or continuous) depends on the value of another variable. Interaction plots plot both variables together on the same graph. 
# 
# While these plots can help us interpret the interaction effects, we would normally use hypothesis tests to determine whether the effect is statistically significant. We also have to be aware that these plots can display random sample error rather than an actual effect, specially in small samples. 
# 
# - Parallel lines: No interaction occurs.
# - Nonparallel lines: An interaction occurs. The more nonparallel the lines are, the greater the strength of the interaction.

# **Interaction: Sex-smoker**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.sex, response=df.charges, trace=df.smoker,
                                   ax=ax, plottype='both', colors=['red','blue'],markers=['D','^'], ms=8)


# - We see that smoking has a huge positive relationship with medical costs. 
# - On the other hand the effect of sex seems almost non-existent, for both smokers and non smokers. 

# In[ ]:


px.histogram(df, x="charges",nbins=int(np.sqrt(len(df.charges))),color='smoker',
             template='ggplot2',facet_col='smoker', 
             facet_row='sex').update_traces(dict(marker_line_width=1,marker_line_color="black"))


# - The smoker distribution is very spread out for both males and females.

# **Interaction: Age - Smoker**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.age, response=df.charges, trace=df.smoker,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# - Age seems to have a positive relationship with medical costs, for both smokers and non-smokers alike, however for smokers the relationship is less evident.

# **Interaction Plot : Bmi - Smoker**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.bmi, response=df.charges, trace=df.smoker,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# In[ ]:


px.scatter(df, x="bmi", y="charges", color="smoker",trendline='ols',
           template='ggplot2',facet_col='sex').update_traces(dict(marker_line_width=1, marker_line_color="black"))


# - We see a very strong relationship between smokers with higher bmi with medical charges. (The R-squred is 0.71 in the case of females smokers.!)
# - For smokers, when their bmi > 30; their medical costs seem to skyrocket. 
# - The relationship between non smokers, bmi and medical charges seems to follow a white gaussian noise.
# - We can use this insight to engineer binned discrete dummy variables to integratates the interaction of having a high BMI levels and being a smoker.

# **Interaction: Children - Smoker**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.children, response=df.charges, trace=df.smoker,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# **Interaction: Sex-Children**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.children, response=df.charges, trace=df.sex,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# - From up to to the third child, charges tend to grow for both sexes, however from the 4th children on, average charges drop drastically. (We have have to take these results with a grain of salt, though, because we have very few observations with 4 or 5 children.

# **Interaction: Region-Sex-Smoker**

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.region, response=df.charges, trace=df.sex,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
fig = sm.graphics.interaction_plot(x=df.region, response=df.charges, trace=df.smoker,
                                   ax=ax, plottype='both', colors=['blue','red'],markers=['D','^'], ms=8)


# - We don't see any clear regional trend with medical costs, however, males from the southeast seem to be higher charged and females from the shouthwest less. 

# ### 1.4 EDA conclusions
# 
# - Smoking seems to be by far the most significant feature to determine the charges. There is a very small difference between sexes and smoking. 
# 
# - Age has a positive relationship with charges. The relationships of non smokers is more consistent than of non smokers.
# 
# - There is a very strong positive relationship between bmi-smoker(yes) and higher medical charges. (more in the case of females than males).
# 
# - For smokers, with bmi > 30; the medical costs seem to skyrocket. 
#       
# - For non smokers over 42 years old, there is a strong consistency in having higher medical costs. 
#      
# - Parents of 4 children or more and no smokers tend to have much lower medical costs on average than otherwhise. (very few of those observations though).

# # 2. Feature Engineering
# 
# Feature engineering is the process of transforming data into features to act as inputs for machine learning models such that good quality features help in improving the overall model performance. Using the insights derived in the previous chapter we can generate new features to reproduce the significant interactions detected and make our linear model more flexible and powerful. 
# 
# As we saw from our EDA analysis, the interactions between our features seem to have a lot of explanatory to our target. We didn't explore the dependencies of our higher order polynomials, however with Polynomial Features will test the model accuracy up to the 5th Polynomial with cross-validation.

# **One-Hot-Encoding (Dummy Variables)**
# 
# We can represent our categorical variables using the one-hotencoding or one-out-of-N encoding, also known as dummy variables. The idea behind dummy variables is to replace a categorical variable with one or more new features
# that can have the values 0 and 1. 

# In[ ]:


df2=df.copy()
X=df2.iloc[:,0:5]
y=df2.iloc[:,6]
X=pd.get_dummies(X)


# In[ ]:


X.head()


# **Quantile-based Binning and Discretization**
# 
# In order to make linear models more powerful on continuous data we can use binning (also known as discretization) of the feature to split it up into multiple new features which compile significant interactions and increase the expressiveness of our feature matrix.
# 
# The problem of only using continuous features in linear regression is that often the distribution of values in these features is skewed. Binning, also known as quantization is used for transforming continuous numeric features into categories. Each bin represents a specific intensity and hence a specific range of continuous numeric values fall into it. 
# 
# We can use quantile-based binning to avoind ending up with irregular bins which, not uniform based on the number of values which fall in each bin.

# We engineer new dummy variables for the interaction of BMI and smoking. (5 bins)

# In[ ]:


X['bmi_bins_smoker_yes'] = pd.cut(x=X['bmi']*X.smoker_yes, bins=X['bmi'].quantile([0, .125, .250, .375, .5, .625, .75, .875, 1.]))


# We engineer new dummy variables for the interaction of age and not smoking. (3 bins)

# In[ ]:


X['age_bins_smoker_no'] = pd.cut(x=X['age']*X.smoker_no, bins=X['age'].quantile([0, .25, .5, .75, 1.]))


# We engineer a new dummy variable for the interaction of having 4 children or more and not smoking.

# In[ ]:


X['+4children_smoker_no'] = pd.cut(x=X['children']*X.smoker_no, bins=[0,3,6])


# In[ ]:


X=pd.get_dummies(X)


# **Interactions, Polynomials and logarithms.**

# In[ ]:


X['age*bmi']=X.age*X.bmi
X['age*smoker_yes']=X.age*X.smoker_no
X['bmi*smoker_no']=X.bmi*X.smoker_yes
X['bmi**2']=X.bmi**2
X['log_age']=np.log(X.age)
X['log_bmi']=np.log(X.bmi)


# ### 2.2 Our feature engineered Matrix X

# In[ ]:


X.head()


# # 3. Linear Regression Cross Validation and model Selection

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


# In[ ]:


#cross-validation function for N-K(2,110,10) fold cross-validation
def cross_val(model,features,predictor):
    RMSE=[]
    model = lm()
    for KFold in range (2,110,10):
        scores = cross_val_score(model, X=features, y=predictor, cv=KFold, scoring='neg_mean_squared_error')
        scores=-scores
        rsme_score=np.sqrt(scores).mean()
        RMSE.append(rsme_score)
        return np.mean(RMSE)


# ### 3.1 Root Mean square error RMSE of original features, feature engineered matrix and interactions & higher order polynomials.

# In[ ]:


RMSE=[]


# **RMSE with Original feature Matrix :**

# In[ ]:


oX=X.iloc[:,0:6]
RMSE.append(cross_val(lm,oX,y))


# **RMSE with our feature engineered Matrix :**

# In[ ]:


RMSE.append(cross_val(lm,X,y))


# **3.2 Interactions & higher order polynomials**
# 
# Interactions only -> If true, only interaction features are produced: 
# features that are products of at most ``degree`` *distinct* input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``,etc.).
# 
# Interactions only -> If False, generate a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form
# ``d[a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].``

# In[ ]:


for i in range (2,6):
    RMSE.append(cross_val(lm,PolynomialFeatures(i,interaction_only=True).fit_transform(oX),y))


# In[ ]:


for i in range (2,6):
    RMSE.append(cross_val(lm,PolynomialFeatures(i,interaction_only=False).fit_transform(oX),y))


# ### 3.3 Root Mean Square error (RMSE) with different Matrix features

# In[ ]:


fig = go.Figure(data=go.Scatter(x=['Original X','Feature engi X','Inter. d2 X','Inter. d3 X ',
                           'Inter. d4 X','Inter. d5 X','PolynomialF.d2','PolynomialF.d3',
                           'PolynomialF.d4','PolynomialF.d5'], y=RMSE))
# Edit the layout
fig.update_layout(title='Root Mean Square error (RMSE) with different Matrix features',
                   xaxis_title='Feature Matrix',
                   yaxis_title='RMSE')
fig.show()


# Our feature Engineered feature Matrix reproduces a smaller RMSE than all the other cross-validated models. 

# In[ ]:


model = lm().fit(X,y)
residuals=y-model.predict(X)
df['residuals']=residuals
df['prediction']=model.predict(X)
df['index']=df.index


# ### 3.4 Model Summary

# In[ ]:


# Fit and summarize OLS model
mod = sm.OLS(y, X)
result = mod.fit()
print(result.summary())


# In[ ]:


prediction_summary=result.get_prediction(X).summary_frame()


# ### 3.5 Error Analysis
# 
# When trying to solve a new machine learning problem, one common approach is to build a quick basic learning model and iterate on it. Identifying patterns in the errors and keep on fixing them. Manually examining the errors our model is systematically making, can give you insights into what to do next. This process is called error analysis. 

# **Residual Plots**

# In[ ]:



# Residual Plots
def regression_residual_plots(model_fit, dependent_var, data, size = [10,10]):
    """
    This function requires:
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
    
    Arguments:
    model_fit: It takes a fitted model as input.
        Obtainable through Statsmodels regression: 
            model_fit = sm.OLS(endog= DEPENDENT VARIABLE, exog= INDEPENDENT VARIABLE).fit()
    dependent_var: string of the pandas column used as the model dependent variable.
    data: pandas dataset where the dependent variable is located. The model data.
    size: default [10,10]. Updates the [width, height], inputed in matplotlibs figsize = [10,10]
        
    Ive only run it on simple, non-robust, ordinary least squares models,
    but these metrics are standard for linear models.
    """
    
    # Extract relevant regression output for plotting
    # fitted values (need a constant term for intercept)
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    ########################################################################
    # Plot Size
    fig = plt.figure(figsize=size)
    
    # Residual vs. Fitted
    ax = fig.add_subplot(2, 2, 1) # Top Left
    sns.residplot(model_fitted_y, dependent_var, data=data, 
                              lowess=True, 
                              scatter_kws={'alpha': 0.5}, 
                              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
                 ax=ax)
    ax.set_title('Residuals vs Fitted')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')

    # Annotations of Outliers
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        ax.annotate(i, xy=(model_fitted_y[i], model_residuals[i]));

    ########################################################################
    # Normal Q-Q
    ax = fig.add_subplot(2, 2, 2) # Top Right
    QQ = sm.ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax)
    ax.set_title('Normal Q-Q')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')

    # Annotations of Outliers
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        ax.annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                model_norm_residuals[i]));

    ########################################################################
    # Scale-Location Plot
    ax = fig.add_subplot(2, 2, 3) # Bottom Left
    plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8}, ax=ax)
    ax.set_title('Scale-Location')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('$\sqrt{|Standardized Residuals|}$');
    # Annotations of Outliers
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        ax.annotate(i, 
                                   xy=(model_fitted_y[i], 
                                       model_norm_residuals_abs_sqrt[i]));

    ########################################################################  
    # Cook's Distance Plot
    ax = fig.add_subplot(2, 2, 4) # Bottom Right
    plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals, 
                scatter=False, 
                ci=False, 
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
               ax=ax)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(-3, 5)
    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')

    # Annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        ax.annotate(i, xy=(model_leverage[i],model_norm_residuals[i]))

    # Shenanigans for Cook's distance contours
    def graph(formula, x_range, label=None):
        x = x_range
        y = formula(x)
        plt.plot(x, y, label=label, lw=1, ls='--', color='red')
    p = len(model_fit.params) # number of model parameters
    graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50), 
          'Cook\'s distance') # 0.5 line
    graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
          np.linspace(0.001, 0.200, 50)) # 1 line
    plt.legend(loc='upper right')
    plt.savefig('residual_plots.png',bbox_inches='tight')
    plt.show()

print("Residual Plots Function Ready")


# Source code for Residual Plots, Authored by Emre Can here:
# 
# https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034

# In[ ]:


regression_residual_plots(result, dependent_var="charges",data=df)


# In[ ]:


px.scatter(df, x='index', y='residuals',title="Model Residuals",trendline='lowess').update_traces(dict(marker_line_width=1,
                                                                                    marker_line_color="pink"))


# In[ ]:


px.histogram(df, x="residuals", marginal="box",template='ggplot2',nbins=int(np.sqrt(len(df.residuals))),
             title="Model Residuals histogram and box-plot").update_traces(dict(marker_line_width=1, marker_line_color="black"))


# Our model seem to underpredict many outlier observations.

# In[ ]:


px.scatter(df, x='index', y='residuals',color='smoker',
           title="Model Residuals with Smoker yes/no").update_traces(dict(marker_line_width=1,marker_line_color="black"))


# Seems that most extreme residuals are of some non smokers with very high medical costs. Our dataset doesn't bring enough explanatory power to predict more accurately these cases. 

# In[ ]:


px.scatter(df, x='prediction',y='charges',trendline='ols',
           title="Predicted values vs Actual Values ").update_traces(dict(marker_line_width=1,marker_line_color="black"))


# In[ ]:


df['squared_pearson_residuals']=result.resid_pearson**2
px.scatter(df, x='prediction',y='charges',trendline='lowess',
           title="Predicted values vs Actual Values - Bubble Size = Squared person residuals",
           color='smoker',size='squared_pearson_residuals').update_traces(dict(marker_line_width=1,marker_line_color="black"))


# As we have identified, most extreme residuals seem to be non smokers with very high medical costs. (red bubbles)

# ### 3.5 Influential Observations: Outlier pruining to derive better generalizability ?

# In[ ]:


influence= result.get_influence().summary_frame()
df['cook_d']=influence.cooks_d


# Cook's distance tries to identify the points that have more influence than other points. Such influential points tend to have a considerable impact on the inclination of the regression line. In other words, adding or removing those points from the model can completely change the model's statistics. Cook's Distance depends on both, the residuals and the leverage, hat value. It summarizes directly how much all the fitted values change when the i-th observation is eliminated. A data point that has a large D. of Cook indicates that the data point strongly influences the fitted values.

# **Cook's distance of our observations**

# In[ ]:


influence = result.get_influence()
#c is the distance and p is p-value
(c, p) = influence.cooks_distance
fig, ax = plt.subplots(figsize=(15,5))
fig=plt.stem(np.arange(len(c)), c, markerfmt=",")


# In[ ]:


df['cook_d']=df['cook_d'].fillna(0)


# We can generate a second linear regression model omiting the values with a cook's distance higher than a certain treshhold.

# In[ ]:


px.scatter(df, x='prediction',y='charges',trendline='ols',
           title="Predicted values vs Actual Values bubble size = cook's distance",
           size='cook_d',color='smoker').update_traces(dict(marker_line_width=1,marker_line_color="black"))


# In[ ]:


df_o=df[df['cook_d'] < 0.005]


# In[ ]:


len(df)-len(df_o)


# **We have "pruned" the 49 observations with the highest Cook's distance.**

# In[ ]:


cook_d=df[df['cook_d'] > 0.005]
px.scatter(cook_d, x='prediction',y='charges',trendline='lowess',
           title="Predicted values vs Actual Values PRUNED OBSERVATIONS bubble size = cook's distance",
           size='cook_d',color='smoker').update_traces(dict(marker_line_width=1,marker_line_color="black"))


# We see a clear separation in the the "pruned" observations between smokers and non smokers.

# In[ ]:


X=df_o.iloc[:,0:6]
y=df_o.iloc[:,6]
X=pd.get_dummies(X)


# We proceed to build a simple feature Matrix :

# In[ ]:


X['bmi_bins_smoker_yes'] = pd.cut(x=X['bmi']*X.smoker_yes, bins=[0,20,30,40,50,60])


# In[ ]:


X['age_bins_smoker_no'] = pd.cut(x=X['age']*X.smoker_no, bins=[0,42,70])


# In[ ]:


X['+4children_smoker_no'] = pd.cut(x=X['children']*X.smoker_no, bins=[0,3,6])


# In[ ]:


X=pd.get_dummies(X)


# We fit the model with the new "pruned" data :

# In[ ]:


# Fit and summarize OLS model
mod = sm.OLS(y, X)
result = mod.fit()
print(result.summary())


# In[ ]:


df_o['residuals']=y-result.predict(X)
df_o['prediction']=result.predict(X)
df_o['squared_residuals']=df_o['residuals']**2


# In[ ]:


(cross_val(lm,X,y))


# In[ ]:


px.scatter(df_o, x='prediction',y='charges',trendline='ols',
           title="Predicted values vs Actual Values model without obvervations with Cook's d. > 0.005 - bubble size = squared residuals",size='squared_residuals').update_traces(dict(marker_line_width=1,marker_line_color="black",marker_color='violet'))


# In[ ]:


px.scatter(df_o, x='prediction',y='charges',trendline='ols',color='smoker',
           title="Predicted values vs Actual Values model without obvervations with Cook's d. > 0.005 - bubble size = squared residuals",
           size='squared_residuals').update_traces(dict(marker_line_width=1,marker_line_color="black"))


# ### Conclusions

# Thanks to the omission of the 49 observations with the highest Cook's distance, we have been able to increase the accuracy of oure cross-validated linear regression model to an R-squared of 0.905. and RMSE to 3233.18. This means that our algorithm can make reasonably good predictions. 
# 
# Pruning data is that it's a complex process and should be executed with caution. One must be careful about being too aggressive while pruning outliers. 

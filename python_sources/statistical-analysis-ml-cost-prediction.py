#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sns.set_style('ticks')


# In[60]:


df = pd.read_csv('../input/insurance.csv')


# # EDA Section

# ## Region Distribution

# In[61]:


f,ax = plt.subplots(figsize=(10,5))
sns.countplot(x='region', data=df, palette="hls",orient='v',ax=ax,edgecolor='0.2')
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.3, i.get_height()+3,             str(round((i.get_height()/df.region.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax.set_xlabel("Region", fontsize=13)
ax.tick_params(length=3,labelsize=12,labelcolor='black')
ax.set_title("Region Distribution", fontsize=14)
x_axis = ax.axes.get_yaxis().set_visible(False)
sns.despine(left=True)
plt.show()


# ## Age Distribution by Categories

# In[62]:


#Let classify age into 4 well known categories, which are 
#'Adolescent',"Young Adult","Adult","Senior"
cut_points = [17,20,35,50,65]
label_names = ['Adolescent',"Young Adult","Adult","Senior"]
df["age_cat"] = pd.cut(df["age"],cut_points,labels=label_names)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
sns.countplot(x='age_cat',data=df,palette='Greens_r',orient='v',ax=ax1,edgecolor='0.2')
for i in ax1.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax1.text(i.get_x()+0.3, i.get_height()+3,             str(round((i.get_height()/df.age_cat.shape[0])*100))+'%', fontsize=12,
                color='black') 
ax2.hist('age', bins=10,data=df,edgecolor='0.2')
ax1.set_xlabel("Age Categories", fontsize=13)
ax1.tick_params(length=3,labelsize=12,labelcolor='black')
ax1.set_title("Age Distribution by Categories", fontsize=14)
ax2.set_xlabel('Age',fontsize=13)
ax2.set_title('Age Distribution',fontsize=14)
x_axis = ax1.axes.get_yaxis().set_visible(False)

f.subplots_adjust(wspace=0.22,right=1.5)
sns.despine(left=True)
plt.show()


# ## Age Distribution by Gender

# In[63]:


def gender_dist_plot(x_val,title):
    f,ax = plt.subplots(figsize=(10,5))
    sns.countplot(x=x_val, data=df, palette=['dodgerblue','lightpink'],hue='sex',hue_order=['male','female'],
                  orient='v',ax=ax,edgecolor='0.2')
    for i in ax.patches:
        ax.text(i.get_x()+0.1, i.get_height()+3,                 str(round((i.get_height()/df.region.shape[0])*100))+'%', fontsize=11,
                    color='black') 
    ax.set_xlabel(title, fontsize=12,color='black')
    ax.tick_params(length=3,labelsize=12,labelcolor='black')
    ax.set_title(title +' Distribution by Gender', fontsize=13)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=[1,0.9],fontsize=12,title='Gender Type',ncol=2)
    sns.despine(left=True)
    return plt.show()

gender_dist_plot('age_cat','Age Category')


# ## Region Distribution by Gender

# In[64]:


gender_dist_plot('region','Region')


# In[65]:


## Region Distribution by Male Smoker
male_data = df[df.sex=='male']
female_data = df[df.sex=='female']

def sex_dist(data,gender,title_color):    
    f,ax = plt.subplots(figsize=(10,5))
    sns.countplot(x='region', data=data, palette=['ForestGreen','saddlebrown'],hue='smoker',
                  hue_order=['no','yes'],orient='v',ax=ax,edgecolor='0.2')
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+0.1, i.get_height()+3,                 str(round((i.get_height()/data.region.shape[0])*100))+'%', fontsize=12,
                    color='black') 
    ax.set_xlabel("Region", fontsize=13)
    ax.tick_params(length=3,labelsize=12,labelcolor='black')
    ax.set_title('Region Distribution by '+ gender +' Smoker', fontsize=14,color=title_color)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    ax.legend(loc=[1,0.9],fontsize=12,title='Smoker type')
    sns.despine(left=True)
    return plt.show()

sex_dist(male_data,'Male','blue')


# ## Region Distribution by Female Smoker

# In[23]:


sex_dist(female_data,'Female','purple')


# ### Check ... if BMI is Normality Distributed

# In[66]:


from scipy import stats
from scipy.stats import norm, skew,kurtosis

def data_transform(data,input):
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(8,8))
    #plots
    sns.boxplot(x=input, data = data,ax=ax1,orient='v')
    sns.distplot(data[input],ax=ax2,color='blue',hist=False)
    res = stats.probplot(data[input], plot=ax3)

    axes = [ax1,ax2]
    kwargs = {'fontsize':14,'color':'black'}
    #for i in range(len(axes)):
        #x_axis = axes[i].axes.get_yaxis().set_visible(False)
    ax1.set_title(input+' Boxplot Analysis',**kwargs)
    ax1.set_xlabel('Box',**kwargs)
    ax1.set_ylabel('BMI Values',**kwargs)

    ax2.set_title(input+' Distribution',**kwargs)
    ax2.set_xlabel(input+' values',**kwargs)

    ax3.set_title('Probability Plot',**kwargs)
    ax3.set_xlabel('Theoretical Quantiles',**kwargs)
    ax3.set_ylabel('Ordered Values',**kwargs)

    f.subplots_adjust(wspace=0.22,right= 2)
    sns.despine()
    return plt.show()

data_transform(df,'bmi')


# **Comment:** The technique used to check for normality is the probability plot that derive from the QQ-plot. 
# By observing the probability plot, we can see that the data fit the line almost perfectly except on the two tails. the data on the tails are a little bit above the line, which means the tails are a little longer. The longer tails are usually due to Outliers. I have found a good article online that explained the concept better. Here it is http://data.library.virginia.edu/understanding-q-q-plots/ 

# ## Categorize BMI value 

# Let create some classification group from the BMI values.
# we will consider:
#     1. Underweight if bmi value is between 14 - 18.99
#     2. Normal if bmi value is btw 19 - 24.99
#     3. Overweight if bmi value is btw 25 - 29.99
#     4. Obese if bmi value is above 30

# In[67]:


cut_points = [14,19,25,30,65]
label_names = ['Underweight',"normal","overweight","obese"]
df["bmi_cat"] = pd.cut(df['bmi'],cut_points,labels=label_names)
gender_dist_plot('bmi_cat','BMI')


# ## Charges Feature Analysis

# Let visualize the Charge feature to see how it is distributed. It will be better for our model if the charge feature is kinda normally distributed or have few outliers

# In[68]:


data_transform(df,'charges')


# From the analysis above, we can observed the following:
#     1. The charge feature is Not normally distributed 
#     2. VERY Skew to the left. 
#     3. The Charge distribution is heavily affected by OUTLIERS.
# To solve the issue mention above, we will use the natural Log transformation on the Charge to reduce outliers and  skweness

# In[69]:


df.charges = np.log1p(df.charges)
data_transform(df,'charges')


# # Machine Learning Section

# This problem is a Multivariate Linear Regression. I am going to approach this technique in two different ways:
#         1. Using Linear Regression 
#         2. Using Other Machine Learning method

# ### Scatter Plot Analysis

# In[70]:


def scatter_analysis(hue_type,palette,data):
    sns.lmplot(x = 'bmi',y='charges',hue=hue_type,data=data,palette=palette,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
    plt.title('Scatterplot Analysis',fontsize=14)
    plt.xlabel('BMI',fontsize=12)
    plt.ylabel('Charge',fontsize=12)
    plt.legend(loc=[1.1,0.5],title = hue_type, fontsize=13)
plt.show()
scatter_analysis('smoker',['ForestGreen','saddlebrown'],df)


# **Smoker scatter plot Comment** 
# From The graph above 
#     1. Smoker are charged generally more than Non smokers 
#     2. There is an increase in charge amount for Smoker as their bmi values increase. which imply a linear relationship in the case of smoker 
#     3. But for non-smoker, the charge tend to be inconsistent no matter bmi value. 

# ### Correlation Analysis

# In[84]:


plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df.corr(),annot=True,robust=True)
plt.title('Correlation Analysis on the Dataset',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()


# ** Heatmap Comment** The analysis on the dataset as a whole only show a strong correlation with the Age but we know from the scatter plot above that they may be a correlation with bmi if you are a smoker. So, we will split the dataset into two parts as mention above and performs the analysis on each category

# ## Part 1: Smoker Dataset Analysis

# In[71]:


#Let drop all categorical variable create during the EDA Analysis
df.drop(['age_cat','bmi_cat'],axis=1,inplace=True)
##Split the data into smoker dataset and non-smoker dataset
df_smoker = df[df.smoker=='yes']
# Convert all categorical columns in the dataset to Numerical for the Analysis
df_smoker = pd.get_dummies(df_smoker,drop_first=True)
from scipy.stats import pearsonr


# ## Statistical Analysis

# Early above, the scatter plot was indicating that there is a relationship between the bmi values and the charges. Two analytic methods (Correlation and p_value analysis) will be performed to determine if the relationship can be proven statistically. 

# #### correlation Analysis

# In[35]:


plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()


# There is a strong Correlation between the bmi and age parameters and charges. However there are no correlation between others parameters and Charges. 

# #### p_value Analysis

# In[36]:


#p_value Analysis
p_value = [round(pearsonr(df_smoker['charges'],df_smoker[i])[1],4) for i in df_smoker.columns]
pvalue_table = pd.DataFrame(p_value,df_smoker.columns).reset_index()
pvalue_table.columns=['colmuns_name','p_value']
pvalue_table.sort_values('p_value')


# ** pValue Comment ** if we set our threshold of p_value = 0.05, which mean a p_value below 0.05 will be statistically significant,
# The Table above shows that age, bmi and region_southeast are statistically significant and in their relationship with the charges.
# Now, we have to check weather the parameters which are statistically significant ARE NOT CORRELATED among each other 
# if we check back on the correlation graph, we observed that [bmi - region_southeast] have a correlation score of **0.27**, [age - bmi] **0.06**, and [age - region_southeast] **0.062**. 
# Due to a high correlation between [bmi - region_southeast] but a low correlation between [region_southeast - charges]
# I think any contribution of the region_southeast parameter to the model will be absorbed in bmi parameter.
# **Conclusion** we will drop all the columns with low p_value(less than 0.05) including region_southeast

# ####  Scatter plot Analysis for smoker

# In[72]:


df_smoker.drop(['children','sex_male', 'region_northwest',
       'region_southeast', 'region_southwest'],axis=1,inplace=True)
scatter_analysis(None,['ForestGreen','saddlebrown'],df_smoker)


# Now we know this age and bmi have a relationship with charges and from the scatter plot, that relationship seems to be linear. Therefore we will use a linear model for the machine leaning analysis also called **Multivariate Linear Regression**. The model used for our prediction will be of the form 
#         **y_predict = intercept + cte1 x age + cte2 x bmi**
#         where intercept, cte_1 and cte_2 are all constant that we will be trying to find

# ## Multivariate Linear Regression Analysis for Smoker

# In[45]:


from sklearn.metrics import explained_variance_score,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


# #### Multivariate Model built & Coefficient

# In[73]:


X = df_smoker.drop('charges',axis=1)
y = df_smoker['charges']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
#Standardizing the values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
## Build  & Evaluate our Model
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('intercept: {:.4f} \ncte1: {:.4f} \ncte2: {:.4f}'.format(model.intercept_,model.coef_[0],model.coef_[1]))

print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,y_test),np.sqrt(mean_squared_error(y_pred,y_test))))


# The model accuracy score being about ** 0.7**  it isn't that bad for a linear regression. the loss is **less than 0.2**.
# Now let visualize the predicted values vs. testvalues on a scatter plot. What we expect is to see the point group together and forming a linear trend so that a point will have close value from the x and y axis.

# #### Linear Regression Visualization Result for Smoker

# In[74]:


def model_scatter_plot(model):
    title = str(model)
    title = title.split('.')[3]
    title = title.replace("'>",'')
    lreg = model()
    lreg.fit(X_train,y_train)
    y_pred = lreg.predict(X_test)
    #model_table
    model_table = pd.DataFrame(y_pred,y_test).reset_index()
    model_table.columns=['y_test','y_pred']
    #Model Graph
    sns.lmplot(x = 'y_test',y='y_pred',data = model_table,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},fit_reg=True)
    plt.title(title +' Analysis',fontsize=14)
    plt.xlabel('y_test',fontsize=12)
    plt.ylabel('y_pred',fontsize=12)
    #plt.scatter(y_test,y_pred)
    return plt.show()

model_scatter_plot(LinearRegression)


# #### use the Model on real data 

# In[75]:


def model_apply(age,bmi_value):
    ## Exple: for a smoker who is age number with bmi = bmi_value, 
    #how much would he pay for insurance
    c = [[age, bmi_value]]
    #we have to transform the data from the standard scaler
    c = sc.transform(c)
    charge_value = model.coef_[0]*(c[0][0]) + model.coef_[1]*(c[0][1]) + model.intercept_
    charge_value = np.exp(charge_value) 
    x = ('The Insurrance Charges for a {:.1f} years old person who is a Smoker with an bmi = {:.1f} will be {:.4f}'.format(age,bmi_value,charge_value))
    # we use the np.exp() because we transformed the value of charge during the charge EDA earlier above
    return print(x)


# In[76]:


#if you are a smoker of 19 yr old and bmi of 32 then what insurrance would you be charged?
model_apply(19,32)


# ## Explore Other Machine Learning Models

# For these other models, the advantage the have is their flexibility and disadvantage is their Interpretability. In these models we are not looking for coefficient cuz they are None. So, the model will be evaluated base on the R square and minimal loss.  

# #### Models Score visualisation for other Machine Learning methods for smoker

# In[77]:


def robust_model(input):
    #Model type to evaluate
    model_list = [ExtraTreesRegressor(),RandomForestRegressor(),GradientBoostingRegressor(),
            LinearRegression(),xgb.XGBRegressor()]
    r_score = []
    loss = []
    for reg in model_list:
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)
        r_score.append(explained_variance_score(y_pred,y_test))
        loss.append(np.sqrt(mean_squared_error(y_pred,y_test)))
    ## Model score table
    model_str = ['ExtraTrees','Random Forest','Gradient Boosting',
            'Linear Regression','XGB Regressor']
    other_model = pd.DataFrame(r_score,model_str).reset_index()
    other_model.columns = ['Model','R(Square)']
    other_model['loss'] = loss
    other_model.sort_values('R(Square)',ascending=False,inplace=True)
    ## Model Graph
    ax = other_model[['R(Square)','loss']].plot(kind='bar',width=0.7,
                            figsize=(15,7), color=['slategray', 'darkred'], fontsize=13,edgecolor='0.2')
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+.1, i.get_height()+0.01,                 str(round((i.get_height()), 3)), fontsize=12, color='black',)
    ax.set_title('Regression Model Evaluation For '+input,fontsize=14,color='black')
    ax.set_xticklabels(other_model.Model, rotation=0, fontsize=12)
    ax.set_xlabel('Model',**kwargs)
    x_axis = ax.axes.get_yaxis().set_visible(False)
    sns.despine(left=True)
    return plt.show()

robust_model('Smoker')


# #### Models Visualization for other Machine Learning methods

# In[78]:


# Let's visualize the test data vs. the predicted data
model_scatter_plot(ExtraTreesRegressor)

model_scatter_plot(GradientBoostingRegressor)

model_scatter_plot(RandomForestRegressor)


# ## Part 2: Dataset for Non-Smoker

# #### Scatter plot analysis for non-smoker 

# In[79]:


df_non_smoker = df[df.smoker=='no']

scatter_analysis(None,['ForestGreen','saddlebrown'],df_non_smoker)


# #### Heatmap for Non-smoker analysis

# In[80]:


# Convert all categorical columns in the dataset to Numerical for the Analysis
df_non_smoker['children'] = df_non_smoker['children'].astype('category')
df_non_smoker = pd.get_dummies(df_non_smoker,drop_first=True)
#correlation Analysis
plt.figure(figsize=(12,8))
kwargs = {'fontsize':12,'color':'black'}
sns.heatmap(df_non_smoker.corr(),annot=True,robust=True)
plt.title('Correlation Analysis for Smoker',**kwargs)
plt.tick_params(length=3,labelsize=12,color='black')
plt.yticks(rotation=0)
plt.show()


# From the heat map above we can observe that only the age is strongly correlated to the charges

# #### Scatter plot of Age vs. Charges

# In[54]:


# Let plot the age vs. charge scatter plot to see the correlation between them
sns.lmplot(x = 'age',y='charges',data=df_non_smoker,size=6,aspect=1.5,
           scatter_kws={"s": 70, "alpha": 1,'edgecolor':'black'},legend=False,fit_reg=True)
plt.title('Scatterplot Analysis',fontsize=14)
plt.show()


# ### Simple Linear Regression Analysis for Non Smoker

# Because only Age in correlated to charges, we are going to use a simple linear model for this section. So, there will be only one coefficient and intercept that we need to find

# In[81]:


X = df_non_smoker['age']
y = df_non_smoker['charges']
X = X.reshape(-1, 1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
#Standardizing the values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Model fitting
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('intercept: {:.4f} \ncte1: {:.4f}'.format(model.intercept_,model.coef_[0]))

print('Model_Accuracy_Score (R Square): {:.4f} \nLoss(RMSE): {:.4f}'.format(r2_score(y_pred,y_test),np.sqrt(mean_squared_error(y_pred,y_test))))


# #### Linear Regression Visualization Result For Non Smoker

# In[82]:


model_scatter_plot(LinearRegression)


# The scatter plot of the predicted vs. test values for non smoker show that the age distribution need to be transformed to have a better age distribution

# #### Models Score visualisation for other Machine Learning methods for non smoker

# In[83]:


robust_model('Non Smoker')


#  The R^2 for the models are very low which show that the model don't perform well....ie bad model. Also the loss, which is the root mean square error, is very high which imply a wider gap between the predicted value and the true value

# ## Conclusion
# 

# ### Thank you for reach till this point. I will appreciate any feedback ....Please an UPVOTE will be very much appreciated

# In[ ]:





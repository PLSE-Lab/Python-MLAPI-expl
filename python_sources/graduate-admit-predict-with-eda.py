#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Following notebook contains in-depth Exploratory Data Analysis for better understanding of data followed by building and comparing machine learning models. I have also added a custom input section where user can input his/her details and can see the chances of getting an admit.

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


dataset=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# # Exploratory Data Analysis

# * Let's check the column names

# In[ ]:


dataset.columns


# * Also the column types

# In[ ]:


dataset.dtypes


# * Changing the target variable name

# In[ ]:


dataset['Admit_rate']=dataset['Chance of Admit ']


# In[ ]:


dataset=dataset.drop('Chance of Admit ', axis=1)


# Let's see the distribution of the target variable

# In[ ]:


sns.set(style="whitegrid")
sns.distplot(dataset['Admit_rate'])


# 1. It has pretty much normal distribution, now what about outliers

# In[ ]:


sns.boxplot(dataset['Admit_rate'])


# There are some outliers, we will handle them during data pre-processing

# Let's check the relationship of features with target variable

# Sequence will be:
# 1. GRE Score
# 2. TOEFL Score
# 3. University Rating
# 4. SOP
# 5. LOR
# 6. CGPA
# 7. Research
# 
# I have some initial intuition about this data.
# Chances of Admit will increase if we increase the values of following variables:
# 1. GRE Score
# 2. TOEFL Score
# 3. SOP
# 4. LOR
# 5. CGPA
# 6. Research
# 
# As the University rating increases, we can assume that competition will increase between students and hence the chances of Admit will go down. Now we will try to confirm this intuition.

# In[ ]:


sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'])


# So our intution regrading GRE scores was true! As the scores increase, chances of Admit increases as well.

# In[ ]:


sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'])


# Same is the scenario with TOEFL scores

# In[ ]:


sns.barplot(x=dataset['SOP'], y=dataset['Admit_rate'])


# * As the quality of SOP increases, the chances of Admit also increases

# In[ ]:


dataset['LOR']=dataset['LOR ']
dataset=dataset.drop('LOR ', axis=1)


# In[ ]:


sns.barplot(x=dataset['LOR'], y=dataset['Admit_rate'])


# So as the quality of Letter of Recommendation increases, chances of Admit also increases. So far our inuition is correct. Let's check for remaining variables.

# In[ ]:


sns.lineplot(x=dataset['CGPA'], y=dataset['Admit_rate'])


# Overall trend is increasing but there are a lot of fluctuations. It shows that high CGPA does not always mean high chances of Admit.

# In[ ]:


sns.barplot(x=dataset['Research'], y=dataset['Admit_rate'])


# Again, with research experirence, chances are that an aspirant will get an admit from the University.

# In[ ]:


sns.barplot(x=dataset['University Rating'], y=dataset['Admit_rate'])


# * So this variable shows opposite of our expectations. It is because the University rating is of the University in which the aspirant has done his/her Undergraduate degree. Obviously, if you have done your Undergrad from some reputed University, then your chance of Admit will also increase.

# * So far our data has a linear relationship with the target variable. Let's do some complex visualisations to understand the data better

# In[ ]:


sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['Research'])


# Even if you have high GRE scores, having no research experience will definitely be a hinder your chances of getting an admit.

# In[ ]:


palette = sns.color_palette("mako_r", 5)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['University Rating'],palette=palette, legend="full")


# With respect to University Ratings of an aspirant, it affects the chances of admit in following ways:
# 1. For aspirants with low GRE score, higher University Ratings are definitely as plus point.
# 2. As the GRE score increases, the affect of University ratings on the chances of Admit also decreases.
# 
# Based on the insights, it is safe to say that aspirants with lower University ranking should aim for a higher GRE score in order to get an admit.

# In[ ]:


palette = sns.color_palette("mako_r", 9)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['SOP'],  palette=palette,legend="full")


# Following insights can be drawn from this:
# 1. Low rated SOPs combined with Low GRE scores will definitely affect your chances of Admit.
# 2. High GRE scores can cover up for average or even low rated SOPs.
# 
# GRE scores are a lot more important for aspirants who have a weak SOP and University rankings.

# Now an important question. 
# Can high GRE scores cover up for low CGPA and vice-versa? 
# Let's check it out!

# So for better understanding, we will be creating bins for CGPA in following manner:
# 1. Category 1 for High CGPA
# 2. Category 2 for Average CGPA
# 3. Category 3 for low CGPA

# In[ ]:


dataset['CGPA_cat']=0


# In[ ]:


temp = [dataset]


# In[ ]:


for data in temp:
    data.loc[dataset['CGPA']<=8,'CGPA_cat']=3,
    data.loc[(dataset['CGPA']>8) & (dataset['CGPA']<=9),'CGPA_cat']=2,
    data.loc[(dataset['CGPA']>9) & (dataset['CGPA']<=10),'CGPA_cat']=1
    


# In[ ]:


palette = sns.color_palette("mako_r", 3)
sns.lineplot(x=dataset['GRE Score'], y=dataset['Admit_rate'], hue=dataset['CGPA_cat'], palette=palette, legend="full")


# As shown, High GRE scores cannot cover up for high cgpa as the difference between aspirants with same GRE scores but different CGPA is clearly visible.

# TOEFL Scores

# We will be doing a similar kind of analysis for TOEFL scores category.

# In[ ]:


sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['Research'])


# * Aspirant with no research experience is at clear cut disadvantage incomparison to the one having a research background. This insight is useful for those aspirants who are trying to get into universities which do not accept GRE.

# In[ ]:


palette = sns.color_palette("mako_r", 5)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['University Rating'],palette=palette, legend="full")


# Aspirants with low University ratings will have low chances of admit in comparison to the ones from higher ranking Universities. This gap is less when TOEFL scores are less but it increases significantly when TOEFL score increases.

# In[ ]:


palette = sns.color_palette("mako_r", 9)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['SOP'],  palette=palette,legend="full")


# TOEFL scores seems to be covering up for low rated SOPs.

# In[ ]:


palette = sns.color_palette("mako_r", 3)
sns.lineplot(x=dataset['TOEFL Score'], y=dataset['Admit_rate'], hue=dataset['CGPA_cat'],palette=palette, legend="full")


# Higher CGPA students again have a clear cut advantage when compared to aspirants with lower CGPA. Even high TOEFL scores cannot bridge the gap.

# # Data Pre-processing

# Here, we will check for missing values, covert categorical variables(since we have all variables in number for so no need) and do some feature engineering if necessary.

# Let's check for missing values

# In[ ]:


dataset.isnull().sum()


# No missing values. Let's check the correlation between variables and plot a heatmap.

# In[ ]:


dataset2=dataset.drop("Serial No.", axis=1)


# In[ ]:


corr=dataset2.corr()
sns.heatmap(corr, cmap="YlGnBu")


# Let's make a better version of this

# In[ ]:


corr = dataset2.corr()
fig, ax = plt.subplots(figsize=(8, 8))
colormap = sns.diverging_palette(220, 10, as_cmap=True)
dropSelf = np.zeros_like(corr)
dropSelf[np.triu_indices_from(dropSelf)] = True
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)
plt.title('Graduate Admissions - Features Correlations')
plt.show()


# In[ ]:


print (corr['Admit_rate'].sort_values(ascending=False)[:10], '\n') #top 15 values


# CGPA is 88% correlated with Admit chances while CGPA categorical is negatively correlated. It is because I have created bins in reverse order.
# In case you have questions about correlations, click on this link-
# https://www.displayr.com/what-is-correlation/
# 

# In[ ]:


num = [f for f in dataset2.columns if ((dataset2.dtypes[f] != 'object')& (dataset2.dtypes[f]!='bool'))]

nd = pd.melt(dataset2, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')


# Now we will split dataset into train and target variables

# In[ ]:


target=dataset['Admit_rate']


# In[ ]:


drop=['Serial No.', 'Admit_rate','CGPA_cat']
train = dataset.drop(drop, axis=1)


# Now it is time to build model.

# # Model Implementation
# We will be implementing both Tree and non-Tree based models to compare the performance
# 
# ### Tree Based
# 1. XGBoost
# 
# ### Non-Tree Based
# 1. Lasso Regression
# 2. Linear Regression

# # Score
# To measure the performance of our model, we will be calculating a score out of 100 using the formula
# #### Score = 100 * (0 , 1 - root_mean_squared_error(actual values, predicted values)

# In[ ]:


#Now we will split the dataset in the ratio of 75:25 for train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.25, random_state = 0)


# ## XGBoost Regressor

# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

model = XGBRegressor(n_estimators=1000,learning_rate=0.009,n_jobs=-1)
model.fit(X_train,y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
    

y_pred = model.predict(X_test)
print("XGBoost Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("XGBoost Regressor RMSE on testing set: ", rmse(y_test, y_pred))

print("XGBoost Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))


# * Making scatter plot between actual and predicted points to see the model performance visually.

# In[ ]:


sns.scatterplot(y_test,y_pred)


# In[ ]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()


# # Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
  
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# ## Lasso Regression

# In[ ]:


from sklearn.linear_model import Lasso
best_alpha = 0.0099

regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred = regr.predict(X_test)
print("Lasso Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("Lasso Regressor RMSE on testing set: ", rmse(y_test, y_pred))
print("Lasso Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))


# * Making scatter plot between actual and predicted points to see the model performance visually.

# In[ ]:


sns.scatterplot(y_test,y_pred)


# In[ ]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()


# ## Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))

y_pred = lr.predict(X_test)
print("Linear Regressor MSE on testing set: ", mean_squared_error(y_test, y_pred))
print("Linear Regressor RMSE on testing set: ", rmse(y_test, y_pred))
print("Linear Regressor Score: {}".format(100*max(0,1-(rmse(y_test,y_pred)))))


# * Making scatter plot between actual and predicted points to see the model performance visually.

# In[ ]:


sns.scatterplot(y_test,y_pred)


# In[ ]:


x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, y_pred, label="predicted")
plt.title("Chance of Admit Test and predicted data")
plt.legend()
plt.show()


# # Prediction on custom input

# In[ ]:


gre = int(input("Enter GRE Score - "))
tfl = int(input("Enter TOEFL Score - "))
ur = int(input("Enter Your University Ratings(1 being lowest and 5 being highest) - "))
sop = int(input("Enter Your SOP(s) Ratings(1 being lowest and 5 being highest) - "))
cgpa = float(input("Enter Your CGPA(1-10) - "))
rp = int(input("Enter number of research publications under your name - "))
lor = int(input("Enter number of Letter of Recommendations - "))


# In[ ]:


check = {'GRE Score': gre, 'TOEFL Score':tfl,'University Rating':ur,'SOP':sop,'CGPA':cgpa,
         'Research':rp,'LOR':lor}


# In[ ]:


df = pd.DataFrame(check,columns = ['GRE Score','TOEFL Score','University Rating','SOP','CGPA','Research','LOR'],
                 index=[1])


# In[ ]:


chances = (model.predict(df))*100
chances = round(chances[0])


# In[ ]:


print('Your chances of getting an admit - {}%'.format(chances))


# # Conclusion
# 
# With highest score of 93.71, XGBoost is performing the best for this dataset. We can using hyperparameter tuning with the help of GridSearchCV and further use ensemble methods to improve our score.
# Apart from this, I have also tried to make it work on a real life problem of getting an admit predict. User can input his/her details and can see the chances of getting an admit.  

# # Future Works
# The dataset used is of very small size and of a very particular university. Upon increasing the size of dataset, we can get even better results which can help us to tackle the problems of Master's degree aspirants. 

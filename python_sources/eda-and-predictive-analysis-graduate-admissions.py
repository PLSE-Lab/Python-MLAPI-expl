#!/usr/bin/env python
# coding: utf-8

# **Gradute Admission Dataset EDA and Prediction**

# In[ ]:


import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np


# In[ ]:


df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv', sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')


# In[ ]:


df


# In[ ]:


# Dropping the unnecessary iterative variable
df = df.drop("Serial No.", axis=1)


# Column wise lookup for missing values to avoid faulty/inaccurate predictions 

# In[ ]:


df.isnull().sum()


# Count of University ratings for each each rating (1-5), to check if each rated university has a fair representation 

# In[ ]:


sns.countplot("University Rating", data=df)


# We can see that the universities rated 1 are not widely represented in the dataset and hence we cannot be very certain about the results we obtain for the rating = 1 universities

# Checking the impact of research on the likelehood of admissions

# In[ ]:


# Dataframe containing all the enteries of students who have undertaken research
df_research = df[df["Research"] == 1]

# Dataframe containing all the enteries of students who havent undertaken research
df_no_research = df[df["Research"] == 0]

# Getting a distribution of the likelihood of admission for the students who have gotten an admit and who havent 
plt.figure(figsize=(9, 8))
sns.distplot((df_research["Chance of Admit"]), color='g', bins=100, hist_kws={'alpha': 0.4});
sns.distplot((df_no_research["Chance of Admit"]), color='b', bins=100, hist_kws={'alpha': 0.4});


# Having a research background does not always ensure a high likelihood for getting into a university, but it sure does play a significant role in increasing the chances of an admit.

# Getting a plot of correlation between the variables

# In[ ]:


corr = df.corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)


# We see that factors like LOR, SOP and Research are not as strong indicators of getting an admit as CGPA, GRE and TOEFL

# Scatter plot of each of the columns against the likelihood of admission 

# In[ ]:


for i in range(0, len(df.columns), 5):
    sns.pairplot(data=df,
                x_vars=df.columns[i:i+5],
                y_vars=['Chance of Admit'])


# All the variables are linearly correlated with the chance of admission, i.e. as the score in a particular test becomes higher, the chance of admission also increases. 

# Impact of the rating of SOP on the likelihood of admission

# In[ ]:


plt.figure(figsize=(15, 6))
ax = sns.boxplot(x="SOP", y="Chance of Admit", hue="SOP", data=df)


# The figure clearly indicates that a higher rated SOP does play a role in maximising the likelihood of getting an admit, there always are scenarios where a very high rated SOP does not boost the chance of getting and admit. For example, in the above scenario, this could be further analysed by filtering out the high rated SOPs which have a low chance of admittance and what are the factors causing it 

# In[ ]:


SOP_ratings_to_filter =  [4, 4.5, 5]

# We will consider the outliers to be the ones which lie in the lower 10 percentile of the enteries for that rating of the SOP
for i in SOP_ratings_to_filter:
    df_i = df[df["SOP"] == i]
    df_outliers = df_i[(df_i["Chance of Admit"] < df["Chance of Admit"].quantile(0.30))]
    display(df_outliers)


# From the above results we can figure that high rated SOPs can result in low likelihood of admit if there are low scores in the other accompnying domains.  

# Impact of the rating of LOR on the likelihood of admission

# In[ ]:


plt.figure(figsize=(15, 6))
ax = sns.boxplot(x="LOR", y="Chance of Admit", hue="LOR", data=df)


# From the figure, it seems like LOR ratings have less outliers when plotted against the likelihood of admission as compared to LOR ratings. Checking for the outliers in LOR ratings:

# In[ ]:


LOR_ratings_to_filter =  [4, 4.5, 5]

# We will consider the outliers to be the ones which lie in the lower 10 percentile of the enteries for that rating of the SOP
for i in LOR_ratings_to_filter:
    df_i = df[df["LOR"] == i]
    df_outliers = df_i[(df_i["Chance of Admit"] < df["Chance of Admit"].quantile(0.30))]
    display(df_outliers)


# We can see that there are relatively less outliers for higher ratings of LOR than for higher ratings of SOP which is a "loose" indicator of the fact that LOR ratings play a more significant role in increasing the chance of admission than SOP rating (for the higher rated SOPs and LORs). 

# GRE and TOEFL score exhibit high correlation with the chance of admission. Visualising the chance of admission plotted against GRE and TOEFL scores. 

# In[ ]:


g =sns.scatterplot(x="GRE Score", y="TOEFL Score",
              hue="Chance of Admit",
              data=df);
g.set(xscale="log");


# From the above diagram, we can see that a high score in both GRE and TOEFL is necessary to increase the chance of admit in the universities. A lower score in any of the tests could result in decreasing the chance of admission. 

# Doing the same for SOP and LOR ratings

# In[ ]:


g =sns.scatterplot(x="SOP", y="LOR",
              hue="Chance of Admit",
              data=df);
g.set(xscale="log");


# Again, from the above scatter plot, we can conclude that a top rated SOP and a top rated LOR do increase the chance of admit in a University. A lower rating for any of the SOP or LOR could lower the chance of admit. 

# Adding one more column which suggests if a student has gotten an admit in the university or not. For this purpose, we will set the threshold to be the mean of the likelihood of admissions. All the likelihoods above the threshold would indicate that a student has gotten an admit in the university and all the likelihoods below the threshold would mean no admit.

# In[ ]:


mean_liklehood = df["Chance of Admit"].mean()


# In[ ]:


mean_liklehood


# In[ ]:


df["Admit"] = [1 if df["Chance of Admit"][i] > mean_liklehood else 0 for i in range(len(df))]


# Distribution of admits / no admits as exxpected 

# In[ ]:


sns.countplot("Admit", data=df)


# In[ ]:


df


# Getting count of each of the University rating for Admit = 1

# In[ ]:


df_admit = df[df["Admit"] == 1]
sns.countplot("University Rating", data=df_admit)


# Quite an imbalance in the dataset which could lead us to draw faulty conclusions about universities rated 1 and 2. It is better 
# to drop out these rating and not analyse them further

# In[ ]:


df_admit = df_admit[(df_admit["University Rating"] >= 2)]


# In[ ]:


plt.figure(figsize=(15, 7))
ax = sns.boxplot(x="Admit", y="GRE Score", hue="Admit", data=df)


# A very clear distinction line is drawn by the GRE Scores which determines if a candidate gets an admit or does not.

# Filtering results rating wise and getting average scores for each of the tests for each university rating to get an admit 
# 

# In[ ]:


df_stats = pd.DataFrame()
for i in range(3, max(df["University Rating"] + 1)):
    df_curr_rating = df_admit[(df_admit["University Rating"] == i)]
    stats = [df_curr_rating.mean()]
    df_stats = df_stats.append(stats)

df_stats


# Average scores required in each of the tests to get an admit in the respectively rated university

# A better insight would be given by the percentile of each of these average scores from our dataset, i.e. where do these average scores stand as far as their percentile is concerned

# In[ ]:


from scipy import stats

for col in df_stats:
    df_stats[col + "_percentile"] = [0 for i in range(len(df_stats))]
    arr = []
    for i, ent in enumerate(df_stats[col]):
        percentile = stats.percentileofscore(df[col], ent)
        arr.append(percentile)
    df_stats[col + "_percentile"] = arr


# In[ ]:


# df_stats = df_stats.drop(["Admit", "University Rating_percentile", "Research_percentile", "Chance of Admit_percentile", "Admit_percentile"], axis=1)
df_stats


# In[ ]:


# import matplotlib.pyplot as plt
# ax = df_stats[['GRE Score_percentile','TOEFL Score_percentile', 'CGPA_percentile', 'LOR_percentile', 'SOP_percentile']].plot(kind='bar', title ="Percentile Scores", x=df_stats["University Rating"], figsize=(15, 10), legend=True, fontsize=12)
# ax.set_xlabel("University Rating", fontsize=12)
# ax.set_ylabel("Percentile Scores", fontsize=12)
# plt.show()


df_stats.plot(x="University Rating", y=["GRE Score_percentile", "TOEFL Score_percentile", "SOP_percentile", "LOR_percentile", "CGPA_percentile"], kind="bar", figsize=(10,10))


# It can be seen that it is very important to have a good standing of SOP and CGPA in average to get into a rating = 5 
# university. However, to get into a slightly lesser rated university, i.e. rating = 4, we see that the GRE, TOEFL and 
# CGPA scores need to be up to the mark. The SOP needs to be least competitve of all the factors to get into a rating = 3
# universiy, but for universities rated 4 and 5, it is essential to have a top notch SOP. A LOR might not possibly be a very 
# strong distinguishing factor for getting in a university rated 4 or 5. CGPA plays an incremental role in getting an admit.

# Getting feature importance of each of the features

# In[ ]:


y = df_admit["Chance of Admit"] 
X = df_admit.drop(["Chance of Admit", "Admit", "University Rating"], axis=1)


# In[ ]:


names = X.columns
rf = RandomForestRegressor()
rf.fit(X, y)
print("Features sorted by their score:")
arr = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)
print(arr)

sns.barplot(x=[ent[0] for ent in arr], y=[ent[1] for ent in arr])


# From the above observations, it could be concluded that CGPA is the most important parameter in determining the likelihood of admit and research is the least important parameter. 

# Checking if this is any different for top rated Universities. 

# In[ ]:


df_top = df_admit[df_admit["University Rating"] >= 4.0]


# In[ ]:


df_top


# In[ ]:


y_top = df_top["Chance of Admit"] 
X_top = df_top.drop(["Chance of Admit", "Admit", "University Rating"], axis=1)


# In[ ]:


names = X.columns
rf = RandomForestRegressor()
rf.fit(X_top, y_top)
print("Features sorted by their score:")
arr = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True)

print(arr)
sns.barplot(x=[ent[0] for ent in arr], y=[ent[1] for ent in arr])


# The feature importances almost remain the same for top rated universities as well.

# Checking feature importances based on OLS Regression

# In[ ]:


model = sm.OLS(y,X)
results = model.fit()
results.summary()


# From the above results we can see that SOP and LOR are the factors which are the least significant in determining the likelihood of an admit. 

# For the prediction of likelihood, using Random Forest was a better choice because data fed in the random forest model does not need to be scaled and RF can also handle binary data. RF accomodates the parameters in our model because scaling any of the scores/ratings/reasearch would either result in loss of data or improper scaling because all the parameters lie on various different ranges. 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train);


# In[ ]:


predictions = rf.predict(X_test)
errors = abs(predictions - y_test)

mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:





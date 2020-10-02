#!/usr/bin/env python
# coding: utf-8

# <center><h1>Graduate Schools - Are You Game For It?</h1></center>
# <hr>
# ### Executive Summary
# Graduate school can be an important part of many people's life, as it is often the first step (after obtaining a degree) to many bigger and higher opportunities in their chosen field. Here, we understand what are the important factors that give a candidate confidence in their application. From our observation of trends in the dataset, we see that they agree with our hypothesis that applicants feel more confident when their portofolio is stronger. Also, we see that academic strength/qualifications are the most crucial factors, followed by their statement of purpose, letter of recommendation and lastly, research. However, the poor correlation of Research with other factors could be due to the all-or-nothing quality it had and hence we must be careful not to blindly interpret the data. There could be higher correlation if Research was recorded in 'the number of years of research experience' an applicant had, thus translating it from a discrete to a continuous variable. Lastly, in using our model to predict new applicants, we must be aware of the strength of correlation and the presence of any biasness in our dataset. In particular, Chance of Admit was slightly left-skewed and there is no sufficient data points to interpret someone who does not feel as confident.

# ### Problem Statement & Dataset Selected
# 
# This report aims to understand the factors that give people confidence to apply for graduate schools. This topic is interesting to me because I am at the stage of life where many friends around me are looking to do further studies, be it masters or PhD prorgrams. It is a common understanding that having good academic scores would give someone greater confidence to apply to graduate schools, but what about other factors? For example, how much does knowing your purpose to do a masters/PhD (i.e. statement of purpose), strength of letter of recommendation and having prior research experience bolster one's courage to take the step?
# 
# The dataset I have selected is from [Kaggle](https://www.kaggle.com/mohansacharya/graduate-admissions). The .zip file contains two .csv files. I chose to use Admission_Predict_Ver1.1.csv instead because it has more entries (500 compared to 400 in the Admission_Predict.csv file). The dataset was created from an Indian perspective for applications to masters programs in the US. The factors considered are defined as follow:
# 
# - GRE Score, out of 340
# - TOEFL Score, out of 120
# - University Rating, out of 5
# - Statement of Purpose (SOP), out of 5
# - Letter of Recommendation (LOR), out of 5
# - Undergraduate CGPA, out of 10
# - Research, either 0 or 1
# - Chance of Admit, ranging from 0 to 1
# 
# The creator of the dataset mentioned a few points which would be important in our analysis of it:
# 
# - SOP and LOR are mostly randomly assigned. It was also mentioned that a few entries were either the applicants' views about how strong their SOP and LOR are, or derived from other parameters. Nonetheless, maybe we should still be careful to not place too much weightage on these two largely randomly assigned variables.
# - University Rating refers to a person's undergraduate university rating, and not the graduate school he/she is applying to.
# - Chance of Admit was asked to the participants how confident they felt about about getting accepted _before_ the results of their application were known to them. We should keep this mind and not interpret it as the actual application success rate.
# 
# Nevertheless, much can still be learned from the dataset. For instance, you can see how much people with similar variables as you feel about their chances and from there you can gauge how confident you can reasonably feel about your own chances. For ease of analysis, GRE, TOEFL and CGPA will be considered together as academic strength. It is rational to hypothesise that the stronger one's academic strength is, the more confident one should feel about their chances of acceptance into the program. The same should apply also for having prior research experience particularly if the masters program is research-based, for coming from a better undergraduate university, and when he/she knows their SOP and LOR are very compelling for acceptance.

# ### Methodology, Insights and Evaluation

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
num_rows = df.shape[0]
num_unique_serial_nos = df['Serial No.'].nunique()
no_duplicate = num_rows == num_unique_serial_nos
print('No duplicate serial numbers:', no_duplicate)


# In[5]:


df.drop('Serial No.', axis = 1, inplace = True)
df.head()


# Under the Discussion section in the Kaggle page, several users shared that they had issues accessing df['Chance of Admit'] because there is actually a space following 'Admit', i.e. 'Chance of Admit '. Other columns should also be checked for this issue. For standardisation purposes, columns with such spaces will be renamed to remove the space.

# In[6]:


df.columns


# In[7]:


df.rename(columns={'LOR ': 'LOR', 'Chance of Admit ': 'Chance of Admit'}, inplace = True)
df.columns


# Following which, the dataset should be checked for any missing data.

# In[8]:


print('There is missing data:', df.isnull().values.any())


# Next, we will seek to understand the distribution of the different parameters. To do this we must first identify what are the continuous and discrete variables. Continuous variables here would refer to Chance of Admit, CGPA, GRE and TOEFL scores, and we will use kdeplot to understand their distribution. Strength of SOP and LOR logically speaking should also be treated as continuous variables, but perhaps of the way the survey was conducted, they turned out to be discrete variables instead (i.e. 3, 3.5, 4...). Discrete variables would then be the remaining parameters (on top of SOP and LOR), namely university rating and research, and histograms would be used instead.

# In[9]:


print('-----Continuous parameters-----')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))
sns.kdeplot(
    data = df['GRE Score'],
    kernel = 'gau',
    ax = ax1
)

sns.kdeplot(
    data = df['TOEFL Score'],
    kernel = 'gau',
    ax = ax2
)

sns.kdeplot(
    data = df['CGPA'],
    kernel = 'gau',
    ax = ax3
)

sns.kdeplot(
    data = df['Chance of Admit'],
    kernel = 'gau',
    ax = ax4
)

plt.show()


# From the distribution plots above, it seems that other than Chance of Admit, the other parameters appears relatively uniformly distributed. 

# In[10]:


print('GRE Score has a skewness value of', df['GRE Score'].skew())
print('TOEFL Score has a skewness value of', df['TOEFL Score'].skew())
print('CGPA has a skewness value of', df['CGPA'].skew())
print('Chance of Admit has a skewness value of', df['Chance of Admit'].skew())


# Calculating the skewness of the above four parameters, we see that GRE Score, TOEFL Score and CGPA indeed have values quite close to zero, which imply that they are rather symmetrical. On the other hand, Chance of Admit has a larger negative skewness value, hinting at a biasness that is present in the dataset. However, this does not come as a surprise as people who have submitted their application for graduate school would be expected to have some confidence of their chances.

# In[11]:


print('-----Discrete parameters-----')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (10, 10))
sns.distplot(
    df['University Rating'], ax = ax1,
    bins = 5, kde = False
)

sns.distplot(
    df['SOP'], ax = ax2,
    bins = 9, kde = False
)

sns.distplot(
    df['LOR'], ax = ax3,
    bins = 9, kde = False
)

sns.distplot(
    df['Research'], ax = ax4,
    bins = 2, kde = False
)

plt.show()


# In the Problem Statement section, it was mentioned that SOP and LOR were mostly randomly assigned. However, looking at their histogram plots, the two parameters look far from being random. As with the 'Chance of Admit' parameter, they tended towards having a left-sknewness, which also makes sense since people who are applying for graduate school would have been confident of their SOP and would have seeked to get recommendation letters from professors whom they know would give a more positive or compelling LOR.

# After looking at the distribution of each individual parameters, it is then interesting to look into the how each parameter correlate with one another. To this end, a correlation matrix was plotted.

# In[12]:


corr_mat = df.corr()
mask = np.zeros_like(corr_mat, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr_mat, mask = mask, cmap = 'YlGnBu',
           square = True, linewidth = .5, cbar_kws = {'shrink': .5})

plt.show()


# From the correlation matrix, it appears that 'Research' has the weakest correlation with the other variables, which means that having prior research experience would not help us to predict other paramters of a person. In particular, it was surprising to me that 'Research' correlated with 'LOR' the least. One would think that if you had prior research experience, a professor would then know you better at a personal level to fully understand if you're a good fit for the research-based rigor for graduate school and help you to write a stronger LOR. Perhaps participants did not seek to get LOR from professors they did research under as assumed. It is also possible that the masters programs referred to here are mainly coursework-based and prior research experience would not be entirely helpful. Alternatively, it could also be the case that the education system in India is different from that in Singapore, such that research experiences are not as valued.
# 
# In addition, we can also note that 'Chance of Admit' correlated most strongly with CGPA, GRE and TOEFL scores, which corroborated on our hypothesis. This reinforces the notion that people tend to feel more confident when they are more academically inclined. Therefore, such observations are logically reasonable.
# 
# It is also important to note that CGPA, GRE and TOEFL are strongly correlated with one another. While this is not surprising, since someone with higher CGPA would indicate that they are more academically inclined and/or diligent to do well in GRE and TOEFL, it begs the question whether schools need to look at so many parameters for graduate school application. Also, since GRE and TOEFL have strong correlation with one another, maybe it is reasonable to consider doing away with the TOEFL requirement in the future. In the GRE general test, two out of three sections serve to also test one's command of English, namely verbal reasoning and analytical writing. Therefore, it is not unexpected for GRE Score to correlate strongly with TOEFL Score. Perhaps some form of redundacy exists between two tests that can be removed for greater efficiency and to minimise students' stress in preparation for graduate school.

# Next, we will look more in detail at the relationship between 'Chance of Admit' and the other parameters through regplot analysis.

# In[13]:


plt.figure(figsize = (10, 20))
index = 0

for col in df.columns:
    if col == 'Chance of Admit':
        pass
    else:
        index += 1
        plt.subplot(4, 2, index)
        sns.regplot(
            x = df['Chance of Admit'],
            y = df[col]
        )

plt.show()


# From ploting of the regplots, it is easy to understand why Chance of Admit had the weakest correlation with Research. Compared to other discrete variables (i.e. University Rating, SOP and LOR), Research had a all-or-nothing quality and hence it is impossible for data points to cluster closely around the regression line. Nevertheless, from the plot we observe that when confidence of admission is low, there tended to be more people with no research experience and less people with research experience. The reverse is seen at high confidence of acceptance. This tells us that there should be higher correlation between research and other parameters (not just Chance of Admit), but such information is lost due to the nature of the Research parameter.

# After understanding the different variables in the dataset and not finding any serious inconsistencies, we can then proceed to perform train a regression model to predict a new student's Chance of Admit from the other parameters.

# In[14]:


X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = df['Chance of Admit']


# In[15]:


import statsmodels.api as sm
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[16]:


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
trained_model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)


# In[17]:


plt.scatter(y_test, predictions)
plt.title('Comparison of Predictions with y_test')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# In[18]:


from sklearn.metrics import r2_score

print('Variance:', r2_score(y_test, predictions))


# From the linear regression model, we obtained a reasonably high r^2 value of 0.77 between the predicted and true Chance of Admit values. However, the r^2 value is admittedly still far from the perfect value of 1.0. Looking at the scatter plot between predictions and true values, we see that at lower True Values/Predictions, the points are less clustered around with each other. This could be due to the slight left-skewness in Chance of Admit as noted above, leading to lesser data points at lower confidence for us to train a better linear regression prediction model.

# ### Conclusion
# 
# Application for graduate schools, be it masters or PhD, is admittedly a stressful process. Having been through it myself and seeing friends getting nervous while waiting for the results of their application, it is sometimes good to know where you stand among other applicants or what are the parameters you should focus on in charting the next step of your career. In analysis of the dataset, we see that most variables agreed with our hypothesis that with a stronger portfolio, applicants generally felt more confident of their chances.
# 
# Nonetheless, we must recognise that there are drawbacks in this dataset and in our model. Firstly, due to the way the survey was conducted, the Research variable had a all-or-nothing characteristic. This thus led to poor correlations with all the other factors. Perhaps the surveyors could have recorded how many years of research experience the applicants had, which would lead to more continuous-like readings. Secondly, as the participants were already generally more confident of their chances, it led to the Chance of Admit being more left-skewed. The model thus probably won't give a good representaion of people who feel less confident of themselves.
# 
# Regardless, the model can still be used to give someone an idea where he/she stand compared to other applicants. To make our predictions more robust, perhaps the university (or universities) in question could release some statistics regarding the true application success rates, which can be used to compare with the applicants' paramters and confidence of acceptance.

# ### Citations
# 
# - The idea to use a for-loop to plot the regplots was obtained from this [Kaggle kernel](https://www.kaggle.com/kralmachine/analyzing-the-graduate-admission-eda-ml)
# - The correlation matrix was learnt from an example from the [Seaborn documentation](https://seaborn.pydata.org/examples/many_pairwise_correlations.html)
# - The linear regression model was learnt from [this website](https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6)

# In[ ]:





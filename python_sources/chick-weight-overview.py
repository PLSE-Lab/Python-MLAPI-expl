#!/usr/bin/env python
# coding: utf-8

# **Description of the Dataset**
# ---
# 
# The dataset was presented in the 1990 book "Analysis of Repeated Measures" by Martin J. Crowder and David J. Hand. It is one of the datasets listed for the R software package, for the nlme package. The CSV file was found on Kaggle.
# 
# https://books.google.com/books/about/Analysis_of_Repeated_Measures.html?id=XsGX6Jgzo-IC
# 
# https://stat.ethz.ch/R-manual/R-patched/library/datasets/html/ChickWeight.html
# 
# https://www.kaggle.com/lsind18/weight-vs-age-of-chicks-on-different-diets
# 
# The data was collected by a nutrition student, in order to test the effect of protein in 4 types of feed on early development of chicks. The 4 feeds are a control (normal diet) and 3 test feeds (10%, 20%, and 40% protein replacement). The assumption is that heavier, larger chicks are a benefit to farmers, and this type of data allows the food industry to advertise the advantages of their feed.
# 
# For example, this detailed web page by Purina Mills is designed to convince farmers to buy their feed, due to its health benefits. They emphasize that "early growth requires the correct balance of nutrients", and the first bullet point of the nutrients in their starter feed is that it has "18% protein". That page does not explain why 18% is a good number. Studies like this are needed to demonstrate why this is an advantage.
# 
# https://www.purinamills.com/chicken-feed/education/detail/what-do-baby-chicks-eat-chick-starter-feed-is-key-for-lifetime-success
# 
# I have not been able to find complete information about the dataset. It is obsolete (1990 textbook), but might be substantially older than that. The Google book web page has a scanned version of the pages with the data, but it does not have the book pages that list the sources, and I do not have the $60 book. However, the publisher of the book mentions: "The book contains information obtained from authentic and highly regarded sources. Reprinted material is quoted with permission and sources are indicated."
# 
# For others (like myself) that are unfamiliar about the nutrition of recently hatched chicks, exploring the data is an opportunity to learn more about the domain of chicken development. The data itself has some unusual features, which the authors of the book did not explore.

# In[ ]:


# Colab seems to randomly crash.
# I've had it crash on loading the CSV file, even though that works normally most of the time.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
import matplotlib.pylab as plt
from scipy import stats
import os # accessing directory structure

df1 = pd.read_csv('/kaggle/input/weight-vs-age-of-chicks-on-different-diets/ChickWeight.csv', delimiter=',')
df1.dataframeName = 'ChickWeight.csv'
df1.columns.values[0] = "row_id"
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
# df1.head()  for some reason, this line is causing the notebook to crash without errors reported?


# The study started with 50 chicks (id numbers range from 1-50). Weight measurements were taken every two days (Time = 0 to Time = 20) with an additional measurement on day 21. By day 21 there are only 45 measurements, which accounts for why there are only 578 rows instead of 600.

# In[ ]:


df1.groupby('Time').agg(
    min_weight=('weight', min),
    max_weight=('weight', max),
    avg_weight=('weight', 'mean'),
    num_chicks=('Chick', 'count')    
)

# Uncomment to get number of measurements per chick
# df1.groupby('Chick').agg(num_chicks=('Time', 'count'))


# If we exclude the 5 chicks that do not have day 21 measurements, the other 45 chicks have measurements for all days of the study.
# 
# (Four of the five excluded chicks were from diet 1, but for now we will simply ignore those 5 chicks.)

# In[ ]:


# exclude those 5 chicks from the sample
bad_chicks = [8, 15, 16, 18, 44]
good_chicks = [num for num in range(1,51) if num not in bad_chicks]
df2 = df1[df1.Chick.isin(good_chicks)]
#df2 = df1[~df1.Chick.isin(bad_chicks)]

df3 = df2[df2['Time'] == 21]
df3.groupby('Diet').agg(
    min_weight=('weight', min),
    max_weight=('weight', max),
    avg_weight=('weight', 'mean'),
    num_chicks=('Chick', 'count')    
)


# Besides the raw numbers, it doesn't hurt to visualize the chick weight versus time. Each line represents the development of a particular chick, with a plot for each diet.

# In[ ]:


fig, axs = plt.subplots(1, 4)
fig.set_size_inches(14.0, 4.0)
axs[0].set_title('Diet 1')
axs[1].set_title('Diet 2')
axs[2].set_title('Diet 3')
axs[3].set_title('Diet 4')

cdiet1 = df2[df2.Diet == 1]
cdiet2 = df2[df2.Diet == 2]
cdiet3 = df2[df2.Diet == 3]
cdiet4 = df2[df2.Diet == 4]
for id in good_chicks:
    axs[0].plot(cdiet1[cdiet1['Chick']==id].Time,cdiet1[cdiet1['Chick']==id].weight,label=id)
    axs[1].plot(cdiet2[cdiet2['Chick']==id].Time,cdiet2[cdiet2['Chick']==id].weight,label=id)
    axs[2].plot(cdiet3[cdiet3['Chick']==id].Time,cdiet3[cdiet3['Chick']==id].weight,label=id)
    axs[3].plot(cdiet4[cdiet4['Chick']==id].Time,cdiet4[cdiet4['Chick']==id].weight,label=id)

for i in range(4):
    axs[i].set_xlabel("day")
    axs[i].set_ylabel("weight")
#    axs[i].legend(loc='best')


# The statistics in each sample is limited, but we will assume the distributions are normal. The (absolute value) skewness and kurtosis are both less than 1 for all 4 diets. The Shapiro-Wilk test for normality does not show any exceedingly small p-values.

# In[ ]:


for i in range(1,5):
    print('Diet',i,stats.describe(df3[df3['Diet'] == i].weight))
    print('  S-W normality test ',stats.shapiro(df3[df3['Diet'] == i].weight))

fig, axs = plt.subplots(1, 4)
fig.set_size_inches(12.0, 4.0)
axs[0].hist(df3[df3['Diet'] == 1].weight)
axs[0].set_title('Diet 1')
axs[1].hist(df3[df3['Diet'] == 2].weight)
axs[1].set_title('Diet 2')
axs[2].hist(df3[df3['Diet'] == 3].weight)
axs[2].set_title('Diet 3')
axs[3].hist(df3[df3['Diet'] == 4].weight)
axs[3].set_title('Diet 4')

for i in range(4):
    axs[i].set_xlim(75.0, 405.0)


# We will use Tukey's Honest Significance Differences (HSD) test to check which of the test diets differ from the control.
# 
# We see that all 3 test diets have chicks with a higher mean day 21 weight than the control feed, but with the limited statistics diets 2 and 4 are not significantly different than the control.
# 
# The 3rd diet is significantly different than the control diet, with the null hypothesis rejected with p < 0.5%.

# In[ ]:


df3.boxplot(by ='Diet', column =['weight'], grid = False) 

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog = df3['weight'],      # Data
                          groups = df3['Diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()

# If the distributions were judged to not be normal, then we'd use stats.kruskal to test the null hypothesis


# The length of time of the study affects the results. For days 2-12, diet 4 has a larger difference and the null hypothesis can be rejected. On day 14, none of the diets have p < 5%. From days 16-21, diet 3 has the statistical significance.
# 
# Keep in mind that all of the protein-supplemented diets have a large difference of the means compared to the control, but the significance is limited by the uncertainty due to the study size.
# 
# Looking at the line plots above, the 4th diet is very tightly clustered early on, and has a higher average (at that time) compared to the other diets.

# In[ ]:


df4 = df2[df2['Time'] == 10]

df4.boxplot(by ='Diet', column =['weight'], grid = False) 

tukey = pairwise_tukeyhsd(endog = df4['weight'],      # Data
                          groups = df4['Diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()


# We are already excluding five of the chicks dropped out of the study. The common trait is that the chick weight was constant for several days, before the measurements stopped. Even for the chicks used in this analysis, some of the chicks appear to have their development arrested. For instance, in diet 2, there is an outlier chick whose weight did not increase after day 8. For diet 4, the heaviest chick on day 12 had its weight plateau afterwards.
# 
# While I can't be certain without more information about the original study, I suspect this was caused a common parasite that still affects chickens. The symptoms include lethargy, loss of appetite, and even rapid weight loss.
# 
# https://morningchores.com/coccidiosis-in-chickens/
# 
# With the limited statistics, any chicks becoming ill will add abnormally low weight measurements, that are independent from the effects of the diet. I am curious what will change in the study if we exclude those chicks that stopped getting heavier by day 18 (cluster of measurements around 0).

# In[ ]:


pivot = df2.pivot_table(index=['Diet','Chick'], columns=['Time'], values=['weight']).reset_index()
pdataf = pd.DataFrame(pivot.to_records())
pdataf.columns = [hdr.replace("('weight', ", "ti").replace(")", "")                      for hdr in pdataf.columns]
pdataf['late_diff'] = pdataf['ti21'] - pdataf['ti18']
pdataf.columns.values[1] = "diet"
pdataf.columns.values[2] = "chick"
pdataf.late_diff.hist(bins=30)


# If we require the chick's weight to increase by more than 6 grams from day 18 to day 21, that leaves 12 chicks in the control diet (out of the original 20), and 8 chicks for each of the test diets (out of the original 10 each).

# In[ ]:


df5 = pdataf[pdataf['late_diff'] > 6.0]
df5.groupby('diet').agg(
    min_weight=('ti21', min),
    max_weight=('ti21', max),
    avg_weight=('ti21', 'mean'),
    num_chicks=('chick', 'count')    
)


# This does not change the conclusion that diet 3 (with 20% protein replacement) is significantly better than the control diet. On the other hand, diets 2 and 4 look very similar with each other.

# In[ ]:


df5.boxplot(by ='diet', column =['ti21'], grid = False) 

tukey = pairwise_tukeyhsd(endog = df5['ti21'],      # Data
                          groups = df5['diet'],   # Groups
                          alpha=0.05)         # Significance level
tukey.summary()


# The results of this study indicate why Purina (for example) is using feed with protein replacement ~20%, as there is a benefit in larger chicks by day 21 after hatching.
# 
# If a study of this type was repeated in the future, the 2 most important changes would be:
# 
# 
# 1.   Increased number of chicks, to reduce the effect of outliers
# 2.   Have all the chicks vaccinated, to remove the effect of sick/infected chicks from the study
# 
# 
# It would also be interesting if there was an additional study group that had diet 4 for the first ten days, and then switched to diet 3, to see if that produces heavier growth than just diet 3.

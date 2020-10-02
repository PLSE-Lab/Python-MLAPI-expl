#!/usr/bin/env python
# coding: utf-8

# # Analysis of Advertising Data

# # Introduction
# The goals of this project are to deeply explore data to do with advertising, perform quantitive analysis and achieve predicitons from the data using machine learning techniques.
# The table below describes features of the data.
# 
# |            Feature            |                        Description                        |
# |:-----------------------------:|:---------------------------------------------------------:|
# | 1. Daily Time Spent on a Site | Time spent by the user on a site in minutes.              |
# | 2. Age                        | Customer's age in terms of years.                         |
# | 3. Area Income                | Average income of geographical area of consumer.          |
# | 4. Daily Internet Usage       | Avgerage minutes in a day consumer is on the internet.    |
# | 5. Ad Topic Line              | Headline of the advertisement.                            |
# | 6. City                       | City of the consumer.                                     |
# | 7. Male                       | Whether or not a consumer was male.                       |
# | 8. Country                    | Country of the consumer.                                  |
# | 9. Timestamp                  | Time at which user clicked on an Ad or the closed window. |
# | 10. Clicked on Ad             | 0 or 1 is indicated clicking on an Ad.                    |

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import time

# Linear Algebra
import numpy as np

# Data Processing
import pandas as pd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Stats
from scipy import stats

# Algorithms
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# String matching
import difflib

# Set random seed for reproducibility
np.random.seed(0)

# Stop unnecessary Seaborn warnings
import warnings
warnings.filterwarnings('ignore')
sns.set()  # Stylises graphs


# In[ ]:


df = pd.read_csv('../input/advertising/advertising.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# ## Null Values
# Firstly, let us see if there are any null values within the data set.

# In[ ]:


df[df.columns].isnull().sum() * 100 / df.shape[0]


# Observing the output of this code shows us that there is a 0% presence of null values within each column of the data.

# ## Duplicates
# Next, let us test if the data contains any duplicates.

# In[ ]:


df.duplicated().sum()


# The data is looking all good on this front, there are 0 duplicates.

# ## Categorizing Quantitative and Qualitative Variables
# Here we shall seperate the quantitivate and qualititave variables in order to summerise them. This should give us a bigger picture of what is going on.
# 
# ### Qualitative

# In[ ]:


qual_cols = set(df.select_dtypes(include = ['object']).columns)
print(f'Qualitative Variables: {qual_cols}')


# What is intresting to note here is that Pandas classifies the Timestamp variable as being an object. Hence, it appears in this section here. Clearly a time stamp is not qualitative data.

# In[ ]:


qual_cols = qual_cols - {'Timestamp'}
print(f'Qualitative Variables: {qual_cols}')


# In[ ]:


df[qual_cols].describe()


# Suprisingly there are 237 contries present in the data. Even more suprising is that France appears the most at 9 times. This must mean that the contries in the dataset varies widely and that one contry is not overly present in the data. All the ad topic lines and cities essentially are different for every instance.

# ### Quantitative 

# In[ ]:


quant_cols = set(df.columns) - set(qual_cols)
print(f'Quantitative Variables: {quant_cols}')


# In[ ]:


df[quant_cols].describe()


# # Investigating the Country Variable
# It seems that without feature engineering the city and contry have very little prediction power. This is because they seem to be widley nonhomogeneous. Let us confirm this however.

# In[ ]:


pd.crosstab(df['Country'], columns='count').sort_values('count', ascending=False).head(10)


# In[ ]:


pd.crosstab(df['Country'], df['Clicked on Ad']).sort_values(1, ascending=False).head(10)


# These outputs suggest that no one country has an overall share in the data. 

# # Feature Engineering
# The data has variables that can be turned into more usful features. This section aims to explore this.
# 
# ## Datetime Engineering
# Here we shall attempt to convert the timestamps given to months and days of the week etc.

# In[ ]:


# Convert Timestamp to a more appropreate data type
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Month'] = df['Timestamp'].dt.month
df['Day'] = df['Timestamp'].dt.day
df['Hour'] = df['Timestamp'].dt.hour
df['Weekday'] = df['Timestamp'].dt.dayofweek


# ## Region Mapping
# Let us try can get each contry mapped to a region and continent. We shall import an external dataset.

# In[ ]:


region_df = pd.read_csv('https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv')


# In[ ]:


region_df.head()


# We only really care about the region and sub-region. Let us try to map each country to its respective region and sub-region. Of course not every country will map onto the external dataset properly, we shall have to drop these rows.

# In[ ]:


region_df = region_df[['name', 'region', 'sub-region']]


# In[ ]:


def label_region(row):
    # Match all rows that directly equal one another
    matches = region_df[region_df['name'] == row['Country']]

    if matches.empty:
        # If no matches, check if all csv countries contain the country from a given row 
        matches = region_df[[row['Country'] in country for country in region_df['name']]]
    
    if matches.empty:
        # If still no matches, check if all csv countries contain the first word of a country from a given row 
        matches = region_df[
            [
                row['Country'].split(' ')[0] in country
                for country in region_df['name']
            ]
        ]
        
        if len(matches) > 1:
            # If there was more than one match, we're not intrested
            matches = pd.DataFrame()
    
    if matches.empty:
         # If still no matches, fuzzyily get matches
        matches = difflib.get_close_matches(row['Country'], region_df['name'], cutoff=0.8)
        
        if matches:
            return region_df[region_df['name'] == matches[0]][['region', 'sub-region']].iloc[0]
        else:
            matches = pd.DataFrame()

    if not matches.empty:
        return matches[['region', 'sub-region']].iloc[0]
    else:
        return [np.nan, np.nan] 

df[['region', 'sub-region']] = df.apply(label_region, axis=1)


# In[ ]:


df[df.isna().any(axis=1)]


# As you can see, there are still some rows that didn't manage to get matches to regions. These are in a small minority; we can drop these rows. Here is what our data looks like now.

# In[ ]:


df = df.dropna()
df.head()


# # Visualisations

# In[ ]:


plt.figure(figsize=(20, 10))
sns.countplot(x='Clicked on Ad', data=df)
plt.title("The Number of People that Clicked on Ads")
plt.xlabel("Clicked on Ad")
plt.xticks([0, 1], ('False', 'True'))
plt.ylabel("Count")
plt.show()


# The graph above demontrates that the data is roughly 50/50 as to if the user clicked on the ad.

# In[ ]:


plt.figure(figsize=(10, 10))
sns.pairplot(
    df,
    hue ='Clicked on Ad',
    vars=['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'],
    diag_kind='kde',
    palette='bright'
)
plt.show()


# This plot shows us a **huge** amount of info! The most intresting part of this graph are the very obvious differences between people that clicked on adverts and people that didn't. Furthermore all of the variables bare no correlation to each other! Each univariate distrubtion shows clear differences in kurtosis and means between people that clicked on an ad and the ones that did not. All the pairplots also indicate clear clumping between the two groups. People that do not click on ads seem to follow clearer patterns. People that did not click on ads seem to be less tightly clumped. It would seemd that larger kurtosis values could be attributed to people that did not click on an ad, backing up the previous statment. This would need to be backed up in quantitive analysis to be made sure of however.
# 
# T-tests would be a good way around testing this.

# # T-Test & F-Test Between Groups of People that Clicked on Ads
# Before we can apply a t-test, it is a good idea to first check if our data is normally distrubuted. We also need to figure out if there is a difference in variance, then we need to think carefully about whether testing for a difference in the mean values is useful.
# 
# ## Variance

# In[ ]:


analysis_cols = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']

for col in analysis_cols:
    vars_ = []

    for clicked in [0, 1]:
        var = np.var(
            df[df['Clicked on Ad'] == clicked][col],
            ddof=1
        )
        vars_.append(var)

        if clicked:
            print(f'Sample variance for {col} of clicked: {var}')
        else:
            print(f'Sample variance for {col} of non clicked: {var}')
            
        
    print(f'Differences in Variance: {round(abs(vars_[0] - vars_[1]), 2)}\n')


# These differences in variance are **HUGELY** different. There is really no need for an F-Test here. We can quite confidently say that the variances between the two groups of people are diffent. This would explain the scale difference of the two distrubtions.
# 
# Next let us test if the means are different.
# 
# ## Mean

# In[ ]:


for col in analysis_cols:
    means_ = []

    for clicked in [0, 1]:
        mean = np.mean(
            df[df['Clicked on Ad'] == clicked][col]
        )
        means_.append(mean)

        if clicked:
            print(f'Mean for {col} of clicked: {mean}')
        else:
            print(f'Mean for {col} of non clicked: {mean}')
            
        
    print(f'Differences in Mean: {round(abs(means_[0] - means_[1]), 2)}\n')


# From this we can see that the daiy time spent of the site and daily internet usage have very clear differences in mean. Age and area income are not as clear however. T-tests are needed to determine if these groups are different here. First we need to test if these variables are normally distributed.
# 
# ## Testing for Normality

# Null Hypothesis ($H_0$):
# > The data is normally distributed.
# 
# Alternative Hypothesis ($H_1$):
# > The data is not normally distributed.
# 
# Significance Level:
# > $\alpha = 0.05$

# In[ ]:


alpha = 0.05

for col in analysis_cols:

    for clicked in [0, 1]:
        k2, p = stats.normaltest(df[df['Clicked on Ad'] == clicked][col])

        if clicked:
            print(f'Results for clicked {col}:')
        else:
            print(f'Results for nonclicked {col}:')

        print(f'\tStatistic: {k2}')
        print(f'\tpvalue: {p}')
        
        if p < alpha:
            print('The null hypothesis can be rejected.\n')
        else:
            print('The null hypothesis cannot be rejected.\n')


# Strangley, none of the distrubtions are normal! The distribution of daily internet usage of the people that clicked on the ad is astronomically far off from being a normal distribution! Because none of the distributions cannot be assumed to be normal, a non-parametric test would be more robust here. As both samples can be assumed not to be normally distributed and both can be assumed to be indepenent, a Mann-Whitney test is most appropriate here.
# 
# ## Mann-Whitney U Test

# Null Hypothesis ($H_0$):
# > Data of people clicking on ads is distributed the same as people not clicking on ads.
# 
# Alternative Hypothesis ($H_1$):
# > Data of people clicking on ads is distributed the not the same as people not clicking on ads.
# 
# Significance Level:
# > $\alpha = 0.05$

# In[ ]:


alpha = 0.05

for col in analysis_cols:
        clicked = df[df['Clicked on Ad'] == 1][col]
        non_clicked = df[df['Clicked on Ad'] == 0][col]

        w, p = stats.mannwhitneyu(x=clicked, y=non_clicked, alternative='two-sided')
        
        print(f'Results for {col}: ')
        print(f'\tStatistic: {w}')
        print(f'\tpvalue: {p}')
        
        if p < alpha:
            print('The null hypothesis can be rejected.\n')
        else:
            print('The null hypothesis cannot be rejected.\n')


# # Distribution of People Clicking on Ads Conclusion
# 
# These tests cleary show that we can reject the null hypothesis on all counts. All of these distributions are distributed differently to a very high significance level. The suspicion that age and area income weren't as clear cut seems to be true as well. They are many orders of magnitude less confident than daily internet usage and daily time spent on the site. That being said, these tests still show that they are **VERY** clearly different.
# 
# **Because of this, there is a very large confidence that the groups of people that click on ads are different to people that do not.**

# # Correlation Between Variables

# In[ ]:


# Computer correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colourmap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr, mask=mask, cmap=cmap, vmax=.3, center=0,
    square=True, linewidths=.5, cbar_kws={"shrink": .5},
    annot=True
)

ax.set_title('Correlation Heatmap of the Variables')

plt.show()


# Largley, most of the variables are uncorrelated, there some very notible correlations however. With this we can more closely investigate other pairings of variables.

# In[ ]:


# SET DATA 

month_counts = pd.crosstab(df["Clicked on Ad"], df["Month"])

# for i in range(1, 12):
#     if i not in month_counts:
#         month_counts[i] = 0
        
# CREATE BACKGROUND
months = [
    'Jan', 'Feb', 'Mar', 'Apr',
    'May', 'Jun', 'Jul'
]

# Angle of each axis in the plot
angles = [(n / 7) * 2 * np.pi for n in range(8)]  # Seven months in data

subplot_kw = {
    'polar': True
}

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.xticks(angles[:-1], months)
plt.yticks(color="grey", size=7)

# ADD PLOTS

# PLOT 1
month_counts_nonclicked = month_counts.iloc[0].tolist()
month_counts_nonclicked += month_counts_nonclicked[:1]  # Properly loops the circle back

ax.plot(angles, month_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")
ax.fill(angles,  month_counts_nonclicked, alpha=0.1)

# PLOT 2
month_counts_clicked = month_counts.iloc[1].tolist()
month_counts_clicked += month_counts_clicked[:1]  # Properly loops the circle back

ax.plot(angles, month_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")
ax.fill(angles,  month_counts_clicked, 'orange', alpha=0.1)

plt.title("Counts of People that Click and Didn't Click Ads for Each Month")
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()


# Looking at the months spider chart, strangely there are only 7 months within the dataset.

# In[ ]:


# SET DATA 

weekday_counts = pd.crosstab(df["Clicked on Ad"], df["Weekday"])

# CREATE BACKGROUND
weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Angle of each axis in the plot
angles = [(n / 7) * 2 * np.pi for n in range(8)]  # Seven months in data

subplot_kw = {
    'polar': True
}

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.xticks(angles[:-1], weekdays)
plt.yticks(color="grey", size=7)

# ADD PLOTS

# PLOT 1
weekday_counts_nonclicked = weekday_counts.iloc[0].tolist()
weekday_counts_nonclicked += weekday_counts_nonclicked[:1]  # Properly loops the circle back

ax.plot(angles, weekday_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")
ax.fill(angles,  weekday_counts_nonclicked, alpha=0.1)

# PLOT 2
weekday_counts_clicked = weekday_counts.iloc[1].tolist()
weekday_counts_clicked += weekday_counts_clicked[:1]  # Properly loops the circle back

ax.plot(angles, weekday_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")
ax.fill(angles,  weekday_counts_clicked, 'orange', alpha=0.1)

plt.title("Counts of People that Click and Didn't Click Ads for Each Weekday")
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()


# In[ ]:


# SET DATA 

hour_counts = pd.crosstab(df["Clicked on Ad"], df["Hour"])

# CREATE BACKGROUND

# Angle of each axis in the plot
angles = [(n / 24) * 2 * np.pi for n in range(25)]  # Seven months in data

subplot_kw = {
    'polar': True
}

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=subplot_kw)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

plt.xticks(angles[:-1], list(range(25)))
plt.yticks(color="grey", size=7)

# ADD PLOTS

# PLOT 1
hour_counts_nonclicked = hour_counts.iloc[0].tolist()
hour_counts_nonclicked += hour_counts_nonclicked[:1]  # Properly loops the circle back

ax.plot(angles, hour_counts_nonclicked, linewidth=1, linestyle='solid', label="Didn't Click Ad")
ax.fill(angles,  hour_counts_nonclicked, alpha=0.1)

# PLOT 2
hour_counts_clicked = hour_counts.iloc[1].tolist()
hour_counts_clicked += hour_counts_clicked[:1]  # Properly loops the circle back

ax.plot(angles, hour_counts_clicked, linewidth=1, linestyle='solid', label="Clicked Ad")
ax.fill(angles,  hour_counts_clicked, 'orange', alpha=0.1)

plt.title("Counts of People that Click and Didn't Click Ads for Each Hour")
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()


# Looking at this data, it would seem that at later hours, there are possibly significant people that are not clicking ads. This would be hard reliably test as there between 15 and 25 data entries.  

# In[ ]:


text = ' '.join(topic_line for topic_line in df['Ad Topic Line'])
world_cloud = WordCloud(width=1000, height=1000).generate(text)

plt.figure(figsize=(20, 10))
plt.imshow(world_cloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()


# Unfortunately it looks like it will not be possible to feature engineer groups of phrases or words, based off this word cloud.

# # Outliers
# 
# Here we shall use an IQR method. For any of the quantitivate variables, points greater than 1.5IQR above or below the upper and lower quartiles are assumed outliers.

# In[ ]:


exten_qual_cols = [
    'Daily Time Spent on Site', 'Age',
    'Area Income', 'Daily Internet Usage'
]

outliers_df = pd.DataFrame(columns=df.columns)

for col in exten_qual_cols:
    stat = df[col].describe()
    print(stat)
    IQR = stat['75%'] - stat['25%']
    upper = stat['75%'] + 1.5 * IQR
    lower = stat['25%'] - 1.5 * IQR
    
    outliers = df[(df[col] > upper) | (df[col] < lower)]

    if not outliers.empty:
        print(f'\nOutlier found in: {col}')
        outliers_df = pd.concat([outliers_df, outliers])
    else:
        print(f'\nNo outlier found in: {col}')

    print(f'\nSuspected Outliers Lower Bound: {lower}')
    print(f'Suspected Outliers Upper Bound: {upper}\n\n')

print(f'Number of outlier rows: {len(outliers_df)}')

del outliers


# In[ ]:


outliers_df.head(10)


# This shows that there are 9 people that could be classed as being an outlier. These all belong in the Area Income variable. Because of this, I would not class these as outliers as they could come from areas that have low income. Furthermore, unless the data sourced the area incomes incorrectly or there is a misunderstanding in how that variable was collected/created, there is no need to remove these.

# # Building Models on Data
# 
# ## Prepping the Data
# 
# We shall first encode a few of our variables and create out test and validation data.

# In[ ]:


X = df.copy()

drop_cols = ['Ad Topic Line', 'City', 'Timestamp']

for col in drop_cols:
    X.drop([col], axis=1, inplace=True)


# In[ ]:


encode_cols = ['Country', 'region', 'sub-region']
le = LabelEncoder()

for col in encode_cols:
    X[col] = le.fit_transform(X[col])


# In[ ]:


y = X['Clicked on Ad']
X.drop(['Clicked on Ad'], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

print(f'X Train Shape: {X_train.shape}')
print(f'X Validation Shape: {X_valid.shape}')
print(f'y Train Shape: {y_train.shape}')
print(f'y Validation Shape: {y_valid.shape}')


# # Modelling with Random Forests
# 
# ## Determining the Best Number of Estimators

# In[ ]:


scores = {}

for n_estimators in range(2, 100):
    RF_model = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    RF_model.fit(X_train, y_train)
    RF_predictions = RF_model.predict(X_valid)
    RF_mae = mean_absolute_error(RF_predictions, y_valid)
    scores[n_estimators] = RF_mae


# In[ ]:


plt.figure(figsize=(10, 4))
plt.title("Mean Absolute Error with Number of Estimators of a Random Forest")
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Absolute Error")
plt.plot(scores.keys(), scores.values())
plt.show()


# In[ ]:


best_n_estimators = []

for n_estimators, score in scores.items():
    if score == min(scores.values()):
        best_n_estimators.append(n_estimators)

print(f"Best Number of Estimators: {min(best_n_estimators)}")


# This shows that we get high accuracy with just five estimators in our random forest.

# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=min(best_n_estimators), random_state=0)

rf_time = time.time()
rf_clf.fit(X_train, y_train)
rf_time = time.time() - rf_time

rf_auc = roc_auc_score(y_valid, rf_clf.predict(X_valid))

score_train = rf_clf.score(X_train, y_train)
print('Training Accuracy : ' + str(score_train))

score_valid = rf_clf.score(X_valid, y_valid)
print('Validation Accuracy : ' + str(score_valid))

print()
print(f'AUC: {rf_auc}')
print(f'Time Elapsed: {rf_time} seconds')
print(classification_report(y_valid, rf_clf.predict(X_valid)))


# **classification_report** will tell us the precision, recall value's accuracy, f1 score & support.
# 
# **precision** is the fraction of retrieved values that are relevant to the data. The precision is the ratio of tp / (tp + fp).
# 
# **recall** is the fraction of successfully retrieved values that are relevant to the data. The recall is the ratio of tp / (tp + fn).
# 
# **f1-score** is the harmonic mean of precision and recall. Where an fscore reaches its best value at 1 and worst score at 0.
# 
# **support** is the number of occurrences of each class in y_test.
# 
# With this, it seems that this model **95%** of the time prediected if a person would not click on an ad and **96%** of the time predicted a person would click on an ad.
# 
# **We can try to do better than this though.**

# # Linear Support Vector Classification
# To try and aprove upon the Swiss army knife of classifiers, Random Forests, let us use a classifier that is more refined for the dataset that we have.

# In[ ]:


svc_lin_scores = {}
c = np.linspace(0.0069, 0.0072, 10)

for C in c:
    svc_lin_clf = SVC(random_state=0, kernel='linear', C=C)
    svc_lin_clf.fit(X_train, y_train)
    svc_lin_scores[C] = svc_lin_clf.score(X_train, y_train)


# In[ ]:


plt.figure(figsize=(20, 5))
plt.title("Precision of Linear SVC With Penalty Parameter C")
plt.ylabel("Precision")
plt.xlabel("C")
plt.plot(svc_lin_scores.keys(), svc_lin_scores.values())
plt.show()


# In[ ]:


svc_lin_clf = SVC(random_state=0, kernel='linear', C=0.007, probability=True)

svc_lin_time = time.time()
svc_lin_clf.fit(X_train, y_train)
svc_lin_time = time.time() - svc_lin_time

svc_lin_auc = roc_auc_score(y_valid, svc_lin_clf.predict(X_valid))

score_train = svc_lin_clf.score(X_train, y_train)
print('Training Accuracy : ' + str(score_train))

score_valid = svc_lin_clf.score(X_valid, y_valid)
print('Validation Accuracy : ' + str(score_valid))

print()
print(f'AUC: {svc_lin_auc}')
print(f'Time Elapsed: {svc_lin_time} seconds')
print(classification_report(y_valid, svc_lin_clf.predict(X_valid)))


# This seems like this is as good as the classifier will get, without overfitting. This is because the only parameter we can tune in a linear SVC is the penalty parameter C. Degree only applies to a poly kernal and $\gamma$ only applies to non linear hyperplanes.

# # K Nearest Neighbors

# In[ ]:


knn_scores = {}

for k in range(1, 30):
    knn_clf = KNeighborsClassifier(k)
    knn_clf.fit(X_train, y_train)
    knn_scores[k] = knn_clf.score(X_train, y_train)


# In[ ]:


plt.figure(figsize=(20, 5))
plt.title("Precision of k Nearest Neighbors Classifier With k Nearest Neighbors")
plt.ylabel("Precision")
plt.xlabel("k Nearest Neighbors")
plt.plot(knn_scores.keys(), knn_scores.values())
plt.show()


# Looking at the graph, it would seem that 1-nearest neighbors would be the best. This might lead to overfitting however. For more generalised datasets, it would seem that 3-nearest neighbors would be a safer bet.

# In[ ]:


knn_clf = KNeighborsClassifier(3)

knn_time = time.time()
knn_clf.fit(X_train, y_train)
knn_time = time.time() - knn_time

knn_auc = roc_auc_score(y_valid, knn_clf.predict(X_valid))

score_train = knn_clf.score(X_train, y_train)
print('Training Accuracy : ' + str(score_train))

score_valid = knn_clf.score(X_valid, y_valid)
print('Validation Accuracy : ' + str(score_valid))

print()
print(f'AUC: {knn_auc}')
print(f'Time Elapsed: {knn_time} seconds')
print(classification_report(y_valid, knn_clf.predict(X_valid)))


# This not anywhere near as good as the random forest or support vector classifier from before!

# # Comparing Models

# In[ ]:


rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_valid, rf_clf.predict_proba(X_valid)[:,1])
svc_lin_fpr, svc_lin_tpr, svc_lin_thresholds = roc_curve(y_valid, svc_lin_clf.predict_proba(X_valid)[:,1])
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_valid, knn_clf.predict_proba(X_valid)[:,1])

plt.figure(figsize=(10, 10))

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest AUC: {round(rf_auc, 3)}')
plt.plot(svc_lin_fpr, svc_lin_tpr, label=f'Linear Support Vector Classifier: {round(svc_lin_auc, 3)}')
plt.plot(knn_fpr, knn_tpr, label=f'k-Nearest Neighbors: {round(knn_auc, 3)}')


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([-0.005, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


def pred_time(model, data):
    times = []
    
    for _ in range(1000):
        pred_time = time.time()
        model.predict(data)
        times.append(time.time() - pred_time)

    return np.mean(times)

models = {
    'Random Forest': (rf_clf, rf_time, rf_auc),
    'Linear Support Vector Classifier': (svc_lin_clf, svc_lin_time, svc_lin_auc),
    'k Nearest Neighbors': (knn_clf, knn_time, knn_auc)
}

for name, model in models.items():
    print(f'Model: {name}')
    print(f'Model Fitting Time: {round(model[1], 4)} seconds')
    print(f'Prediction Time: {round(pred_time(model[0], X_valid), 4)} seconds')
    print(f'Model AUC: {round(model[2], 4)}\n\n')


# # Random Forest Variable Importances

# In[ ]:


columns = X.columns
train = pd.DataFrame(np.atleast_2d(X_train), columns=columns)


# In[ ]:


feature_importances = pd.DataFrame(rf_clf.feature_importances_,
                                   index = train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances.head(10)


# In[ ]:


plt.figure(figsize=(13, 7))
sns.barplot(
    x="importance", y='index',
    data=feature_importances[0:10], label="Total"
)
plt.title("Random Forest Variable Importance")
plt.ylabel("Variable")
plt.xlabel("Importance")
plt.show()


# # Modelling Conclusions
# Whilst the random forest could have been tuned further, it had good precision. It did not take too much time to fit the model, which would allow for fast tuning of parameters.
# The linear kernal SVC took a very long time to fit the data. It is faster at predicting than the random forest and k nearest neighbors classifiers. This time taken to fit the data is mittgated, as only one parameter is needed to be tuned.
# The k Nearest Neighbors perfomed the worst in AUC and prediction time. This was not a good model for this data.
# 
# In the end the linear SVC should be used as it had a slightly higher AUC and faster prediction time when compared to the random forest. The end accuracy of this project is **95.4%**.

# # Acknowledgements
# A massive thanks to [Santosh](https://www.kaggle.com/konchada), his work on this dataset with his [notebook](https://www.kaggle.com/konchada/logistic-vs-random-forest-model-for-ad-click) was a big inspiration.

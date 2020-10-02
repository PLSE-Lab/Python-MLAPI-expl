#!/usr/bin/env python
# coding: utf-8

# # Part 1. EDA

# ## Loading libraries and data

# In[ ]:



import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = 'darkgrid') #


import os
print(os.listdir("../input"))
from pylab import rcParams
rcParams['figure.figsize'] = 25, 12.5
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# ## Primary EDA

# In[ ]:


train.info()


# We can conclude, that most of the variables have types of float and integers. Only 5 columns are objects, let's explore which ones

# In[ ]:


categorical = []
numerical = []
for feature in test.columns:
    if test[feature].dtype == object:
        categorical.append(feature)
    else:
        numerical.append(feature)
train[categorical].head()


# So, we have 2 ID columns, the other 3 columns can be converted into integers. Intuitively, "no" can be converted to 0, while "yes" to 1. But for now we won't do that

# Let's explore NA values in data

# In[ ]:


train[numerical].isnull().sum().sort_values(ascending = False).head(8)


# We can actually conclude, that our data is pretty full, only in 5 columns we have observe Missing values. For now it's too early to make conclusion about dropping out first three colums as useful information may be contained there(We will explore the hypothesis using graphs). Moreover, LightGBM and XGBoost are able to handle missing values while training, so that's not a problem.
# 
# We can fill in NAs with mean or median values for the rest 2 cols.

# In[ ]:


test[numerical].isnull().sum().sort_values(ascending = False).head(8)


# Good news is that the same columns with missing values are observed in test! The good thing is to check if we should fill in NAs with mean or median values.

# In[ ]:


train[['meaneduc', 'SQBmeaned']].describe()


# The point is that if we observe outliers in data, we should fill in NAs with median, otherwise it's ok to fill in with mean values. In the table, 50% is the median value, mean is mean :) Here it's fine to use mean values

# In[ ]:


train['meaneduc'].fillna(train['meaneduc'].mean(), inplace = True)
train['SQBmeaned'].fillna(train['SQBmeaned'].mean(), inplace = True)
#the same for test
test['meaneduc'].fillna(test['meaneduc'].mean(), inplace = True)
test['SQBmeaned'].fillna(test['SQBmeaned'].mean(), inplace = True)
train['rez_esc'].fillna(0, inplace = True)
train['v18q1'].fillna(0, inplace = True)
train['v2a1'].fillna(0, inplace = True)


# Other 3 columns we fill in with 0's temporarily

# ## Seaborn

# Seaborn is a great library for visualization. You can choose many palettes, which makes the graphs visually nice. For instance, some of them.

# In[ ]:


sns.set(style = 'darkgrid')
sns_plot = sns.palplot(sns.color_palette('Accent'))
sns_plot = sns.palplot(sns.color_palette('Accent_d'))
sns_plot = sns.palplot(sns.color_palette('CMRmap'))
sns_plot = sns.palplot(sns.color_palette('Set1'))
sns_plot = sns.palplot(sns.color_palette('Set3'))


# In[ ]:


target_values = train['Target'].value_counts()
target_values = pd.DataFrame(target_values)
target_values['Household_type'] = target_values.index
target_values


# Let's map index
# 

# In[ ]:


mappy = {4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"}
target_values['Household_type'] = target_values.Household_type.map(mappy)
target_values


# In[ ]:


sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(15, 8))
ax = sns.barplot(x = 'Household_type', y = 'Target', data = target_values, palette='Accent', ci = None).set_title('Distribution of Poverty in Households')


# As an 'insight', we can claim that classes are skewed. And, probably, without any effort our algorithms will have large errors predicting Extreme Powerty.

# ## Correlations

# In[ ]:


#Let's find out largest correlations and depict them
corrs = train.corr().abs()
corrs1 = corrs.unstack().drop_duplicates()
strongest = corrs1.sort_values(kind="quicksort", ascending = False)
strongest1 = pd.DataFrame(strongest)
temp = strongest1.index.values
first_cols = [i[0] for i in temp]
second_cols = [j[1] for j in temp]
total_cols_corr = list(set(first_cols[:20] + second_cols[:20]))
strongest.head(25)


# In[ ]:


corr = train[total_cols_corr].corr()
sns.set(font_scale=1)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(25, 12.5))
sns.heatmap(corr, cmap=cmap, annot=True, ax=ax, fmt='.2f')


# As a result, we might probably delete some columns without decreasing ROC as they are collinear( for instance, public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC" and coopele, =1 electricity from cooperative, correlation between these two is -.98) 
# 

# In[ ]:


train['v2a11'] = train.v2a1.apply(lambda x: np.log(x+1))
sns.set(font_scale=1, style="darkgrid")
c =  sns.color_palette('spring_d')[4]
sns_jointplot = sns.jointplot('age', 'meaneduc', data=train, kind='kde', color=c, size=6)


# Most of train data is allocated around the age of 20 and mean education of 10 years.
# But we didn't separate data by *Target*.
# Let's do that too.

# In[ ]:


for i in range(1, 5):
    sns.set(font_scale=1, style="white")
    c =  sns.color_palette('spring_d')[i]
    sns_jointplot = sns.jointplot('age', 'meaneduc', data=train[train['Target'] == i], kind='kde', color=c, size=6, stat_func=None)


# The picture became much better. For NonVulnerables both mean years of education and age are higher and allocates around 10 and 20 respectively. It's useful to mention that variance of mean education for NonVulnerables is less than for others.
# 
# For Extreme Poors age in years is the least among these 4 categories.

# ### Living condition comparisons

# Let's check out some of the living conditions for different households

# In[ ]:


def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue = target, aspect = 4, row = row, col = col)
    facet.map(sns.kdeplot, var, shade = True)
    facet.set(xlim = (0, df[var].max()))
    facet.add_legend()
    plt.show()


# In[ ]:


#select some columns
numerical1 = ['v2a11', 'meaneduc', 'overcrowding'] #monthly pay rent, mean education, overcrowd
for numy in numerical1:
    plot_distribution(train, numy, 'Target')
#In the first graph instead of 0's should be nulls(we changed these before). So there is no info about monthly rate payment for non vulnerable households 


# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='Target', y = 'r4h3',ax = ax, data = train, hue = 'Target' )
ax.set_title('Number of men in households', size = 25)


# In[ ]:


f, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='Target', y = 'r4m3',ax = ax, data = train, hue = 'Target' )
ax.set_title('Number of women in households', size = 25)


# The Numbers are close to each other between categories. Probably, these two variables(number of women and men in the household) won't have a large impact on target variable.

# Let's also check the hypothesis that Poorer households have more children

# In[ ]:


ninos = train.groupby(by = 'Target')['hogar_nin', 'Target'].sum()
ninos = pd.DataFrame(ninos)
ninos['mean_children'] = (ninos['hogar_nin']/ninos['Target'])
ninos['Target1'] = ninos.index.map({4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"})
sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(16, 8))
ax = sns.barplot(x = 'Target1', y = 'mean_children', data = ninos, palette='Pastel1', ci = None).set_title('Mean number on children in different households')


# We can observe a huge difference in mean number of children in different types of households, consequently, the hypothesis about the mean number of children in poor households is true. Therefore,  we can create this feature in our data to increase score.

# We explored that Extreme poverty households tend to have more children than nonVulnerables. Let's dig deeper and find out how monthly rate payment per person differ. 
# Firstly, as we have observed a lot of missing values in v2a1(Monthly rent payment),so necessary to say that there will be high bias. 

# In[ ]:


train['v2a1'].replace(0, np.nan, inplace = True)
train["v2a1"] = train.groupby("Target").transform(lambda x: x.fillna(x.median()))
rpd = pd.DataFrame([train['v2a1']/train['hogar_total'], train['Target']]).T
rpd['Target'] = rpd['Target'].map({4: "NonVulnerable", 3: "Moderate Poverty", 2: "Vulnerable", 1: "Extereme Poverty"})
rpd.groupby(by = 'Target').mean()


# In[ ]:


sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(16, 8))
ax = sns.barplot(x = 'Target', y = 'Unnamed 0', data = rpd, palette='Pastel1',order = ["Extereme Poverty","Vulnerable","Moderate Poverty", "NonVulnerable"], ci = None).set_title('Montly rent payment per dweller')


# In[ ]:


#visualization of feature importance of XGB below


valuez = ['meaneduc', 'age', 'qmobilephone','Target', 'r4t3', 'tamhog', 'escolari', 'overcrowding']
tra = pd.melt(train[valuez], "Target", var_name="measurement")
f, ax = plt.subplots()
sns.despine(bottom=True, left=True)
sns.stripplot(x="value", y="measurement", hue="Target",
              data=tra, dodge=True, jitter=True,
              alpha=.05, zorder=1)
sns.pointplot(x="value", y="measurement", hue="Target",
              data=tra, dodge=.532, join=False, palette="dark",
              markers="x", scale=1, ci=None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[4:], labels[4:], title="Target",
          handletextpad=0, columnspacing=1,
          loc="lower right", ncol=1, frameon=True)


# According to the results of XGB and the graph above, algorithms can detect 4th class, but first three classes are difficult to detect as values in variables are almost the same. So the main approach is to generate **killer features**, which will help algorithms to separate first three classes and, therefore, reach 0.5+ F1-score

# .## Multi-Dimensional Reduction and Visualisation with t-SNE

# In[ ]:


from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
X_scaled = scaler1.fit_transform(train[numerical])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne2d = TSNE(random_state=13012)\ntsne_representation2d = tsne2d.fit_transform(X_scaled)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsne3d = TSNE(n_components = 3, random_state = 666)\ntsne_representation3d = tsne3d.fit_transform(X_scaled)')


# In[ ]:


tsne_representation2d = pd.DataFrame(tsne_representation2d, columns = ['First_col', 'Second_col'])
tsne_representation2d['Target'] = train.loc[:, 'Target']


# In[ ]:


tsne_representation3d = pd.DataFrame(tsne_representation3d, columns = ['First_col', 'Second_col', 'Third_col'])
tsne_representation3d['Target'] = train.loc[:, 'Target']


# In[ ]:


sns.set(font_scale=1, style="darkgrid") #CMRmap_r
sns.lmplot( x="First_col", y="Second_col", data=tsne_representation2d, fit_reg=False, hue='Target', legend=False, palette="Set1", size = 17)
plt.legend(loc='lower right')


# Although t-SNE is a bit unstable(changing random state may change the pic.), from the first glance we can claim that there are some clusters, which can help us to separate class 4 from others. Let's look at 3d representation.

# In[ ]:


from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
rcParams['figure.figsize'] = 30, 20
fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(tsne_representation3d.loc[:, 'First_col'], tsne_representation3d.loc[:, 'Second_col'], tsne_representation3d.loc[:, 'Third_col'], s = 29, c = tsne_representation3d.loc[:, 'Target'],
          edgecolors = 'black')
ax.set_title('t-SNE visualization in 3 dimensions', size = 20)
pyplot.show()


# Now we see that a lot of observations of 4-th cluster are detached from others. Unfortunately, I didn't find out how to plot the same with *seaborn* and an attempt to plot it with *plotly* occured to be unsuccessful. If someone has any idea how to plot the same graph(but interactive), you are welcome to share.

# # Part 2. Models

# Let's build a **basic XGBoost model** and take a glance at feature importance 

# In[ ]:


from sklearn.model_selection import train_test_split
import xgboost as xgb


# In[ ]:


y = train['Target']
train = train.drop(['Id', 'Target'] ,axis = 1)
train = train.select_dtypes(exclude=['object'])
test = test.drop('Id',axis = 1)
test = test.select_dtypes(exclude=['object'])


# In[ ]:


y.value_counts()


# here we have a deal with skewed classes, that's why we need to use at least stratification in splitting data

# 

# In[ ]:


y = y - 1
X_train, X_test, y_train, y_test = train_test_split(train, y, stratify = y, test_size = 0.3, random_state = 666)


# In[ ]:


y.value_counts()


# In[ ]:


from sklearn.metrics import f1_score
def evaluate_macroF1(true_value, predictions):  
    pred_labels = predictions.reshape(len(np.unique(true_value)),-1).argmax(axis=0)
    f1 = f1_score(true_value, pred_labels, average='macro')
    return ('macroF1', f1, True) 
params = {
        "objective" : "multi:softmax",
        "metric" : evaluate_macroF1,
        "n_estimators": 100,
        'max_depth' : 9,
        "learning_rate" : 0.23941,
        'max_delta_step': 2,
        'min_child_weight': 9,
        'subsample': 0.72414,
        "seed": 666,
        'num_class': 4,
        'silent': True
    }
xgbtrain = xgb.DMatrix(X_train, label=y_train)
xgbval = xgb.DMatrix(X_test, label=y_test)


watchlist = [(xgbtrain, 'train'), (xgbval, 'valid')]
evals_result = {}
model = xgb.train(params, xgbtrain, 5000, 
                     watchlist,
                    early_stopping_rounds=150, verbose_eval=100)
#we don't need these for now
#xgbtest = xgb.DMatrix(test)
#p_test = model.predict(xgbtest, ntree_limit=model.best_ntree_limit)

#p_test = p_test + 1


# In[ ]:


xgb_fimp=pd.DataFrame(list(model.get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
xgb_fimp1 = xgb_fimp.iloc[0:35]

sns.set(style = 'whitegrid', font_scale=1.4)
fig = plt.subplots(figsize=(10, 15))
ax = sns.barplot(x = 'importance', y = 'feature', data = xgb_fimp1,palette='Accent', ci = None).set_title('Feature importance of XGBooost')


# According to the results, the main feature xgboost extracts is mean education. Then, we observe age, years of education of male head of household squared, overcrowding. 

# In[ ]:


from sklearn.metrics import classification_report
Xgb_test = xgb.DMatrix(X_test)
y_pred = model.predict(Xgb_test, ntree_limit=model.best_ntree_limit)
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# Work in progress... Stay tuned :)

# ## If you like this kernel, please, upvote. It's not that hard for you and, moreover, it motivates me to work and share ideas with you, guys

# 

# # Hope it was informative for all of you. Thanks for your attention!

# In[ ]:





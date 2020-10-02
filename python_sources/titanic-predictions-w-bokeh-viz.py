#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivability Predictions
# 
# In this notebook we will explore the titanic dataset, visualize some important features of the data using the Bokeh module, do some data cleaning and formatting of the data for use in some common classifiers, and finally we will build and evaluate those classifiers on our dataset. 

# # 1 Exploration of Titanic Dataset w/ Bokeh Visuals
# 
# First we explore the dataset to see what features we might want to use for our classifier.  To help with our exploration we practice using the Bokeh visualization module. 

# ## 1.1 Import Packages and Load Data

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb # classifiers using boosted trees

from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import viridis


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## 1.2 Survivability of Males and Females with Age
# 
# For our first explorations we take a look at whether males or females have a better chance of surviving in the training set, and whether or not age seems to make a difference. 

# In[ ]:


train['SexNumerical'] = train.apply(lambda x: 0.0 if x['Sex']=='male' else 1.0, axis = 1)
test['SexNumerical'] = test.apply(lambda x: 0.0 if x['Sex']=='male' else 1.0, axis = 1)
mask = [False if np.isnan(xi) == True else True for xi in train['Age'].values.tolist()]
train = train[mask]
train_male = train[train['Sex']=='male']
train_female = train[train['Sex']=='female']
train_male_age_1 = train_male[train_male['Survived']==1]['Age'].values
train_male_age_0 = train_male[train_male['Survived']==0]['Age'].values
train_female_age_1 = train_female[train_female['Survived']==1]['Age'].values
train_female_age_0 = train_female[train_female['Survived']==0]['Age'].values

hist_male_1, edges_male_1 = np.histogram(train_male_age_1, density = False, bins = range(0,80,5))
hist_male_0, edges_male_0 = np.histogram(train_male_age_0, density = False, bins = range(0,80,5))
hist_female_1, edges_female_1 = np.histogram(train_female_age_1, density = False, bins = range(0,80,5))
hist_female_0, edges_female_0 = np.histogram(train_female_age_0, density = False, bins = range(0,80,5))
xm = np.linspace(min(min(train_male_age_1), min(train_male_age_0)), max(max(train_male_age_0), max(train_male_age_1)), 100)
xm = np.linspace(min(min(train_female_age_1), min(train_female_age_0)), max(max(train_female_age_0), max(train_female_age_1)), 100)


# Calculate survivability overall survivability statistics for males and females.

# In[ ]:


male_survivability = len(train_male_age_1)/len(train_male)
female_survivability = len(train_female_age_1)/len(train_female)
print('Survivability Rates')
print('Males: ' + str(np.round(male_survivability*100., decimals=2)) + '%, Females: '+ str(np.round(female_survivability*100., decimals=2)) + '%')


# In[ ]:


output_notebook()
pm = figure(title = 'Male Survivability with Age', x_range = (0, 85), y_range = (0, 80))
pm.quad(top = hist_male_1, bottom = 0, left = edges_male_1[:-1], right = edges_male_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')
pm.quad(top = hist_male_0, bottom = 0, left = edges_male_0[:-1], right = edges_male_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')
pm.legend.location = 'center_right'
pm.xaxis.axis_label = 'Age'
pm.yaxis.axis_label = 'Individuals'
pf = figure(title = 'Female Survivability with Age', x_range = (0, 85), y_range = (0, 80))
pf.quad(top = hist_female_1, bottom = 0, left = edges_female_1[:-1], right = edges_female_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')
pf.quad(top = hist_female_0, bottom = 0, left = edges_female_0[:-1], right = edges_female_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')
pf.legend.location = 'center_right'
pf.xaxis.axis_label = 'Age'
pf.yaxis.axis_label = 'Individuals'
show(gridplot([pm,pf], ncols = 2, plot_width = 400, plot_height = 400, toolbar_location = None))


# From the age distributions we can immediately see several key takeaways:
# 
# - There were many more males on board than females. 
# - Males had a much lower survivability rate above age 15 than females did.
# - Males aged 15-25 had the lowest survivability rates.  Less than 10% of males in these age groups survived.
# - Females aged 50+ had the highest survivability rates.  Only 10% of females above 50 perished.

# ## 1.3 Effects of Affluence - Ticket Prices and Fare Class
# 
# * First we will look at the price distributions for those that survived and did not. 

# In[ ]:


train_fare_1 = train[train['Survived']==1]['Fare'].dropna()
train_fare_0 = train[train['Survived']==0]['Fare'].dropna()
hist_fare_1, edges_fare_1 = np.histogram(train_fare_1, density = False, bins = range(0,300,10))
hist_fare_0, edges_fare_0 = np.histogram(train_fare_0, density = False, bins = range(0,300,10))

classes = sorted(train['Pclass'].unique())
class_df = pd.DataFrame(index = classes, columns = ['Survivability'])
for c in classes:
    c_tot = train[train['Pclass']==c]
    c_1 = c_tot[c_tot['Survived']==1]
    class_df.at[c, 'Survivability'] = float(len(c_1))/float(len(c_tot))*100


# In[ ]:


output_notebook()
p1 = figure(title = 'Survivability with Fare Amount', x_range = (0, 300), y_range = (0, 200))
p1.quad(top = hist_fare_1, bottom = 0, left = edges_fare_1[:-1], right = edges_fare_1[1:], fill_color = 'navy', alpha = 0.5, legend = 'Survived')
p1.quad(top = hist_fare_0, bottom = 0, left = edges_fare_0[:-1], right = edges_fare_0[1:], fill_color = 'red', alpha = 0.5, legend = 'Perished')
p1.xaxis.axis_label = 'Fare Amount'
p1.yaxis.axis_label = 'Individuals'
p1.legend.location = 'center_right'
cats = ['Class ' + str(x) for x in class_df.index.values.tolist()]

mapper = linear_cmap(field_name='counts', palette=viridis(256) ,low=0. ,high=100.)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))

source = ColumnDataSource(data = dict(cats=cats, counts = class_df['Survivability']))
p2 = figure(title = 'Survivability Rate and Fare Class', x_range = cats, y_range = (0, 100))
p2.vbar(x = 'cats', top = 'counts', color = mapper, width = 0.9, source = source)
p2.yaxis.axis_label = 'Chance of Survival (%)'
p2.add_layout(color_bar, 'right')
show(gridplot([p1,p2], ncols = 2, plot_width = 400, plot_height = 400, toolbar_location = None))


# From this information, we can clearly see that lower fares are associated with lower survivability.   Taking only fare information as an input, we can estimate that passengers who paid less than 50 for their ticket were more likely than not to perish.  Meanwhile passengers who paid more than 50 for their ticket were likely to survive.  This trend can also be seen when looking at the fare class, where first class passengers are nearly three times as likely to survive as third class passengers.

# ## 1.4 Family Size
# 
# We will now move on to the effects of family size.  Some have speculated that by travelling alone, or not travelling alone, one might somehow derive an advantage over others in terms of survivability.  We will add the number of siblings/spouses and the number of parents on board for each passenger into one family size variable and compare survivability rates.

# In[ ]:



train['Family Size']= train['SibSp']+train['Parch']+1
train['Family Size']= train.apply(lambda x: str(x['Family Size']), axis = 1)
test['Family Size']= test['SibSp']+train['Parch']+1
test['Family Size']= test.apply(lambda x: str(x['Family Size']), axis = 1)
train_fs_1 = train[train['Survived'] == 1][['PassengerId','Family Size']].groupby('Family Size').count().reset_index()
train_fs_totals = train[['PassengerId', 'Family Size']].groupby('Family Size').count().reset_index()
train_fs_1 = pd.merge(train_fs_1, train_fs_totals, on = 'Family Size')
train_fs_1['Chance'] = train_fs_1['PassengerId_x']/train_fs_1['PassengerId_y']*100.
source = ColumnDataSource(train_fs_1)

mapper = linear_cmap(field_name='Chance', palette=viridis(256) ,low=0 ,high=100.)
color_bar = ColorBar(color_mapper=mapper['transform'], width=8,  location=(0,0))

output_notebook()
p = figure(title = 'Survivability with Family Size', x_range = train_fs_1['Family Size'], y_range = (0, 100))
p.vbar(x = 'Family Size', top = 'Chance', color = mapper, width = 0.9, source = source)
p.xaxis.axis_label = 'Family Size'
p.yaxis.axis_label = 'Chance of Survival (%)'
p.add_layout(color_bar, 'right')
show(gridplot([p], ncols = 1, plot_width = 400, plot_height = 400, toolbar_location = None))


# From the above we can see that there is a significant preference for average sized families.  Members of families comprising 4 persons fared the best, with a better than a 75% survival rate. Both large families and individuals without families fared poorly.  It's unclear why this relationship exists, but some have speculated that individuals disproportionately perished because families were chosen for lifeboats ahead of individuals.  At the high end, its possible that larger families took longer to organize and therefore experienced lower survival rates.  An alternate theory is that there is a strong correlation between social class and large family sizes.  Its additionally likely that the smaller sample sizes for the higher family sizes play an important role.

# ## 2. Model Building Using XGBoost

# In[ ]:


def train_test_split(X,y, perc):
    '''Performs a simple random sample of the training data returning a training and testing set'''
    trainx = X.sample(frac=perc, replace = False, random_state = 0)
    sel = trainx.index.values.tolist()
    trainy = y.loc[sel]
    notsel = []
    for i in X.index.values.tolist():
        if i not in sel:
            notsel.append(i)
    testx = X.loc[notsel]
    testy = y.loc[notsel]
    return trainx, trainy, testx, testy


# In[ ]:



trainx = train[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()
trainx['Family Size'] = trainx.apply(lambda x: float(x['Family Size']), axis=1)
trainy = train['Survived']
trainx, trainy, testx, testy = train_test_split(trainx, trainy, 0.8)
param = {'max_depth':5, 'eta': 0.5, 'silent': 1, 'booster': 'gbtree', 'objective': 'binary:logistic', 'eval_metric':'error'}
num_round = 10
Dtrain = xgb.DMatrix(trainx, label = trainy)
Dtest = xgb.DMatrix(testx, label = testy)
watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]


# In[ ]:


bst = xgb.train(param, Dtrain, num_round, watchlist)


# In[ ]:


ypred = bst.predict(Dtest)
labels = Dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(ypred)) if int(ypred[i] > 0.5) != labels[i]) / float(len(ypred))))


# In[ ]:


trainx = train[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()
trainx['Family Size'] = trainx.apply(lambda x: float(x['Family Size']), axis=1)
trainy = train['Survived']
Dtrain = xgb.DMatrix(trainx, label = trainy )
bst = xgb.train(param, Dtrain, num_round, watchlist)


# ## Make Predictions on the Evaluation Set

# In[ ]:


xeval = test[['Pclass','Fare','SexNumerical','Age','Family Size']].copy()
xeval['Age']=xeval['Age'].fillna(np.nanmean(xeval['Age']))
xeval['Family Size']=xeval.apply(lambda x: float(x['Family Size']), axis = 1)
xeval['Family Size']=xeval['Family Size'].fillna(np.nanmean(xeval['Family Size']))
xeval['Fare']=xeval['Fare'].fillna(np.nanmean(xeval['Fare']))
Deval = xgb.DMatrix(xeval)
ypred_eval = bst.predict(Deval)
ypred_eval_out = np.hstack([test['PassengerId'].values.reshape(-1,1), ypred_eval.reshape(-1,1)])
ypred_eval_out = pd.DataFrame(ypred_eval_out, columns=['PassengerId', 'Score'])
ypred_eval_out['Survived']=ypred_eval_out.apply(lambda x: 1 if x['Score']>=0.5 else 0, axis =1)
ypred_eval_out['PassengerId']=ypred_eval_out.apply(lambda x: int(x['PassengerId']), axis =1 )
ypred_eval_out = ypred_eval_out[['PassengerId', 'Survived']].set_index('PassengerId')
ypred_eval_out.to_csv('output.csv')


# In[ ]:


ypred_eval.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipe = Pipeline(steps = [('minmax',MinMaxScaler()), ('svc',SVC(C=1., kernel = 'rbf', gamma = 'auto'))])

params = {'svc__C':[0.1,1.0,10.,100., 1000.]}

clf = GridSearchCV(pipe, params, cv=4).fit(trainx, trainy)
print(clf.best_score_, clf.best_params_)


# In[ ]:


svm_predictions = pd.DataFrame(columns = ['PassengerId', 'Survived'], data = np.hstack([test['PassengerId'].values.reshape(-1,1), clf.predict(xeval).reshape(-1,1)])).set_index('PassengerId')
svm_predictions.to_csv('outputsvm.csv')


# In[ ]:





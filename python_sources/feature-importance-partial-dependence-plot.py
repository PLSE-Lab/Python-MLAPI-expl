#!/usr/bin/env python
# coding: utf-8

# ## Feature importance and partial dependence plot in titanic
# In this kernel, I builded both statistical model and machine learning model and explored relationship between features and survival. <br>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Test data is not needed, because it's explanatory analysis. 
titanic = pd.read_csv("../input/train.csv")
titanic.head()


# In[ ]:


titanic.info()


# ### Variable	Definition	Key
# - survival:	Survival	0 = No, 1 = Yes
# - pclass:	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# - sex:	Sex	
# - Age:	Age in years	
# - sibsp:	# of siblings / spouses aboard the Titanic	
# - parch:	# of parents / children aboard the Titanic	
# - ticket:	Ticket number	
# - fare:	Passenger fare	
# - cabin:	Cabin number	
# - embarked:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


# Drop columns : only left basic features
titanic.drop('PassengerId', axis=1, inplace=True)
titanic.drop("Name", axis=1, inplace=True)
titanic.drop("Ticket", axis=1, inplace=True)
titanic.drop("Cabin", axis=1, inplace=True)


# In[ ]:


# Type change
titanic['Pclass'] = titanic['Pclass'].astype(object)


# In[ ]:


# Define categorical and numerical features]
categorical_features = ['Pclass', 'Sex', 'Embarked']
numerical_feature = ['Age', 'Fare']
features = categorical_features + numerical_feature
outcome = 'Survived'


# In[ ]:


# Plot setting
sns.set_context("paper")
sns.set(font='serif')
sns.set_style("white", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})


# ### Missing values

# In[ ]:


total = titanic[features].isnull().sum().sort_values(ascending = False)
percent = (titanic[features].isnull().sum()/titanic[features].isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train_data


# ### Imputation (Age)
# > Compute column mean except zero and impute the missing values.

# In[ ]:


column_means = titanic['Age'].mean(axis=0)
titanic.loc[np.isnan(titanic['Age']), 'Age'] = column_means


# ### Simple visualization

# In[ ]:


for feature in categorical_features : 
    df = titanic.groupby([feature,outcome])[outcome].count().unstack(outcome)
    df.plot(kind='bar', figsize=(10,5))
    plt.title(feature)
    plt.show()


# In[ ]:


for feature in numerical_feature : 
    for value in ['0', '1'] : 
        subset = titanic[titanic[outcome] == int(value)]
        sns.distplot(subset.loc[subset[feature].notnull(), feature], label=value)
        plt.legend()
    plt.show()


# ### Logistic regression

# In[ ]:


# Make dummy variables
titanic_logistic = titanic.copy()
titanic_logistic = titanic_logistic.drop(categorical_features, axis=1)
for feature in categorical_features : 
    dummy = pd.get_dummies(titanic[feature], columns=feature, prefix=feature)
    titanic_logistic = pd.concat([titanic_logistic, dummy.iloc[:,1:]], axis=1)      


# In[ ]:


logistic_features = list(titanic_logistic.columns)
logistic_features = list(set(logistic_features) - set(['Survived']))


# In[ ]:


import statsmodels.api as sm
titanic_logistic['intercept'] = 1
logit = sm.Logit(titanic_logistic[outcome].astype(int), titanic_logistic[logistic_features]) 
result = logit.fit()

# Scipy error fixing..
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

print(result.summary())
print("Odds ratio")
print(np.exp(result.params))


# In[ ]:


oddsratio = np.exp(result.params)
plt.figure(figsize=(8,6))
n_features = len(logistic_features)
plt.barh(range(n_features), oddsratio, align='center')
plt.yticks(np.arange(n_features), logistic_features)
plt.xlabel("Odds ratio")
plt.ylabel("Feature")
plt.ylim(-1, n_features)
plt.show()


# In[ ]:


#  One hot encoding
titanic_onehot = titanic.copy()
titanic_onehot = titanic_onehot.drop(categorical_features, axis=1)
for feature in categorical_features : 
    dummy = pd.get_dummies(titanic[feature], columns=feature, prefix=feature)
    titanic_onehot= pd.concat([titanic_onehot, dummy], axis=1)  


# In[ ]:


onehot_features = list(titanic_onehot.columns)
onehot_features = list(set(titanic_onehot) - set(['Survived']))


# In[ ]:


def plot_feature_importances(model, features):
    plt.figure(figsize=(8,6))
    n_features = len(features)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


# ### Decision Tree Feature importance

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(titanic_onehot[onehot_features], titanic_onehot[outcome])
print("Accuracy on training set: {:.3f}".format(tree.score(titanic_onehot[onehot_features], titanic_onehot[outcome])))
plot_feature_importances(tree, onehot_features)


# ### Random forest feature importance

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf.fit(titanic_onehot[onehot_features], titanic_onehot[outcome])
print('Random Forest - Max Depth = 3')
print("Accuracy on training set: {:.3f}".format(rf.score(titanic_onehot[onehot_features], titanic_onehot[outcome].astype('int'))))
print('Random Forest Feature Importance')
plot_feature_importances(rf, onehot_features)


# ### GradientBoostingTree Feature importance

# In[ ]:


from pandas import read_csv, DataFrame
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

model = GradientBoostingClassifier(n_estimators=20, max_depth=4,learning_rate=0.1, loss='deviance',random_state=1)
model.fit(titanic_onehot[onehot_features], titanic_onehot[outcome])
importances = model.feature_importances_
plot_feature_importances(model, onehot_features)


# ### Partial Dependence Plot
# - When using black box machine learning algorithms like random forest and boosting, it is hard to understand the relations between predictors and model outcome.
# - Partial dependence plot (PDP) is one way to invenstigate relationship between variable and outcome.
# - The partial dependence plot (PDP or PD plot) shows the marginal effect of a feature on the predicted outcome of a previously fit model (J. H. Friedman 2001). 
# - What is partial dependence plot? (https://www.kaggle.com/dansbecker/partial-dependence-plots)
# - python partial dependence plot toolbox (https://github.com/SauceCat/PDPbox)
# 
# _Reference_ 
# - _interpretable machine learning_ (https://christophm.github.io/interpretable-ml-book/pdp.html)

# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)

features_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # For 1, 5 th features, draw interaction plot
fig, axs = plot_partial_dependence(model, titanic_onehot[onehot_features], features_index,feature_names=onehot_features,n_jobs=3, grid_resolution=50)
plt.figure()
plt.subplots_adjust(bottom=0.1, right=1.1, top=1.4) 


# ### Interaction plot : Age * Sex

# In[ ]:


fig = plt.figure()
target_feature = (onehot_features.index("Age"), onehot_features.index("Sex_female"))
pdp, axes = partial_dependence(model, target_feature, X = titanic_onehot[onehot_features], grid_resolution=50)

XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,cmap=plt.cm.BuPu, edgecolor='k')

ax.set_xlabel(onehot_features[target_feature[0]])
ax.set_ylabel(onehot_features[target_feature[1]])
ax.set_zlabel('Partial dependence')

plt.colorbar(surf)
plt.suptitle('Partial dependence of predictors')
                 
plt.subplots_adjust(right=1,top=.9)
plt.show()


# ### 2D Interaction plot

# In[ ]:


from pdpbox import pdp
pdp_inter = pdp.pdp_interact(model,titanic_onehot[onehot_features],features=['Sex_female','Age'], model_features=onehot_features) 
pdp.pdp_interact_plot(pdp_inter, ['Sex_female','Age'], plot_type='grid', x_quantile=True, plot_pdp=True) 


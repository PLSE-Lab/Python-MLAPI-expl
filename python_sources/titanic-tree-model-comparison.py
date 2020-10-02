#!/usr/bin/env python
# coding: utf-8

# # Titanic Tree Model Comparison
# In this notebook, I build a gradient boosting classifier to determine whether someone aboard the Titanic survives or dies. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
train.head()


# # EDA
# I classify the variables based on the data type that they hold. Looking back on the notebook, I would probably split the numerical variables into discrete and continuous categories in the future. (However, this will likely not affect the EDA.)

# In[ ]:


categorical = ['Pclass', 'Sex', 'Embarked']
numerical = ['Age', 'Fare', 'SibSp', 'Parch']
target = 'Survived'


# Descriptive statistics notes:
# * Most passengers are middle-aged adults
# * Age has some missing values
# * Fare is very right-skewed
# * Most passengers have no siblings or spouses on board

# In[ ]:


train[numerical].describe()


# Age is roughly a normal distribution, but Fare is not normal due to high skew and peakedness (kurtosis). Since SibSp and Parch are not continuous variables, I am not too concerned about the normality of their distribution.

# In[ ]:


from scipy.stats import kurtosis, skew

for num in numerical:
    print('{} has kurtosis: {} and skew: {}'
          .format(num, kurtosis(train[num], nan_policy = 'omit'), skew(train[num], nan_policy = 'omit')))


# I plotted the distribution for the numerical variables and visually inspected their normality with the QQ plot. I tried log, sqrt, power transformations to make the distributions more normal, and the best transformation was log for Fare and Age. It is not necessary to transform the data before using a tree-based model, but transformations could help prevent overfitting.

# In[ ]:


from scipy.stats import probplot

for num in numerical:
    f, axs = plt.subplots(1, 3, figsize = (12, 4))
    sns.distplot(train[num], kde = False, 
                 ax = axs[0])
    probplot(train[num], 
             plot = axs[1])
    probplot(np.log1p(train[num]), 
             plot = axs[2])
    axs[1].set_title('No transformation')
    axs[2].set_title('Log transformation')
    plt.tight_layout()
    plt.show()


# I only transform the continuous variables.

# In[ ]:


train['Fare'] = np.log1p(train['Fare'])
train['Age'] = np.log1p(train['Age'])


# Correlation plot notes:
# * Moderate positive correlation between Fare and Parch and SibSp, which are like measures of family size
# * Weak correlation between Age and Parch and SibSp

# In[ ]:


g = sns.heatmap(train[numerical].corr(method = 'spearman'),
                vmax = 0.6, vmin = -0.6,
                annot = True,
                fmt = '.2f',
                square = True)


# In[ ]:


from scipy.stats import ttest_ind

mask = (train[target] == 1)
t_result = []
for num in numerical:
    t_stat, p_val = ttest_ind((train[mask])[num],
                              (train[~mask])[num],
                              equal_var = True,
                              nan_policy = 'omit')
    t_result.append({
        'group_1' : 'Survived',
        'group_2' : 'Died',
        'variable' : num,
        't_stat' : t_stat,
        'p_value' : p_val
    })
t_result = pd.DataFrame(t_result)


# T test between two means notes:
# * There is a significant difference in Age, Fare, and Parch between passengers that survived or died.

# In[ ]:


t_result


# I decided to create a new feature after the first model performed poorly. Feature engineering helped me greatly improve the accuracy of the model. Not shown is a contingency table between Sex and Title, which helped me decide how to group the titles into the four bins: Mr, Mrs_Ms, Master, and Honorable. One important observation from this table is that all the passengers with title "Master" are males, and the majority of these passengers survived even though the overall porbability of survival for males is low.

# In[ ]:


import re

def get_titles(data):
    title_re = re.compile(r'(?:^.+), (\w+)')
    titles = []
    for name in data['Name']:
        titles.append(title_re.findall(name)[0])

    data['Title'] = titles
    #data['Title'].value_counts()

    for i, title in enumerate(data['Title']):
        if title in ['Miss', 'Ms', 'Mlle', 'Mrs', 'Mme']:
            cat = 'Mrs_Ms'
        elif title in ['Mr', 'Don']:
            cat = 'Mr'
        elif title in ['Master']:
            cat = 'Master'
        else:
            cat = 'Honorable'
        data.loc[i, 'Title'] = cat


# In[ ]:


categorical = ['Pclass', 'Sex', 'Embarked', 'Title']
get_titles(train)


# Categorical plot notes:
# * Passengers from Pclass 1 had a higher probability of survival than passengers from Pclass 2 or 3
# * Females are more likely to survive than males
# * Passengers who had the title "Master" were all males but had a higher probability of survival than passengers who had the title "Mr"
# * There are more male than female passengers
# * There are more passengers in Pclass 3 than Pclass 1 or 2
# * The majority of passengers embarked from origin S
# * Passengers who embarked from origin C had a higher probability of survival

# In[ ]:


for cat in categorical:
    g = sns.countplot(x = cat, hue = target, data = train)
    plt.show()


# In[ ]:


g = sns.catplot(x = 'Pclass', hue = target, col = 'Sex', data = train,
                kind = 'count')


# In[ ]:


g = sns.catplot(x = 'Title', hue = target, col = 'Pclass', data = train,
                kind = 'count')


# Chi2 contigency test notes:
# * There is a significant effect of Pclass, Sex, Embarked, and Title on survival.

# In[ ]:


from scipy.stats import chi2_contingency

chi2_result = []
for cat in categorical:
    crosstab = pd.crosstab(train[cat], train[target])
    chi2_stat, p_val, dof, ex = chi2_contingency(crosstab)
    chi2_result.append({
        'var_1' : cat,
        'var_2' : target,
        'chi2' : chi2_stat,
        'dof' : dof,
        'p_value' : p_val
    })
chi2_result = pd.DataFrame(chi2_result)


# In[ ]:


chi2_result


# # Preprocessing
# Check for and fill missing variables

# In[ ]:


train.isnull().sum()


# In[ ]:


X = train.copy()
y = X.pop(target)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

median_imputer = SimpleImputer(strategy = 'median')
mode_imputer = SimpleImputer(strategy = 'most_frequent')
missing_transformer = ColumnTransformer([('num', median_imputer, numerical), 
                                         ('cat', mode_imputer, categorical)])
missing_transformer.fit(X)
X_imp = pd.DataFrame(missing_transformer.transform(X))
X_imp.columns = numerical + categorical
X.drop(numerical + categorical, axis = 1, inplace = True)
X = pd.concat([X, X_imp], axis = 1)


# One-hot encode categorical variables with low cardinality. We can drop the first variable to avoid any collinearity.

# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
X_enc = pd.DataFrame(ordinal_encoder.fit_transform(X[categorical]))
X_enc.columns = categorical
X.drop(categorical, axis = 1, inplace = True)
X = pd.concat([X, X_enc], axis = 1)
X = pd.get_dummies(X, columns = categorical, dtype = np.int64, drop_first = True)


# In[ ]:


X.head()


# Mutual information score notes:
# * Survival is mainly predicted by Title_3.0, Sex_1.0, Fare, and Title_2.0

# In[ ]:


from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X[features], y)
mi_idx = mi.argsort()
for i in mi_idx:
    print('{} has MI score: {}'.format(X[features].columns[i], mi[i]))


# Clustermap notes:
# * There is one cluster with major collinearity between Title_3.0 (Master), Sex_1.0 (Female), and Title_2.0 (Mr). For now, we will not do anything about this because they all have high mutual information score.

# In[ ]:


features = ['Age', 'Fare', 'SibSp', 'Parch', 'Sex_1.0',
            'Embarked_1.0', 'Embarked_2.0', 'Pclass_1.0',
            'Pclass_2.0', 'Title_1.0', 'Title_2.0', 'Title_3.0']
g = sns.clustermap(X.corr(method = 'spearman'))


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y)


# In[ ]:


features = ['Title_2.0', 'Sex_1.0', 'Fare']


# We are going to fit different types of tree classifiers to see which one is the optimal model to predict Survived.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

clfs = {
    'tree' : DecisionTreeClassifier(),
    'rf' : RandomForestClassifier(),
    'extra' : ExtraTreesClassifier(),
    'grad' : GradientBoostingClassifier(),
}


# In[ ]:


from sklearn.metrics import f1_score

clf_result = []
for model_name, model in clfs.items():
    model.fit(X_train[features], y_train)
    preds = model.predict(X_test[features])
    cv_score = cross_val_score(model, X_train[features], y_train, scoring = 'accuracy', cv = 5)
    clf_result.append({
        'model' : model_name,
        'mean acc' : cv_score.mean(),
        'std acc' : cv_score.std()
    })
    
clf_result = pd.DataFrame(clf_result)


# Model selection notes:
# * There does not seem to be much difference between baseline Random Forests or Gradient Boosting Classifiers, but since Gradient Boosting usually performs well for hyperparameter tuning, I will choose this model.

# In[ ]:


clf_result


# # Modeling
# This was the hardest part of the notebook, as I was originally scoring roughly 0.69 with the model that I created because the model was overfitting the training data. I learned a lot about different diagnostic plots and visualizations to assess how well the model is performing in the next section.

# I used a Randomized Grid Search to evaluate different combinations of hyperparameters.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators' : [50, 100, 150, 200],
    'learning_rate' : [0.01, 0.03, 0.05, 0.07, 0.1, 0.3],
    'subsample' : [0.4, 0.45, 0.5, 0.55, 0.7],
    'max_features' : ['auto', 'sqrt'],
    'n_iter_no_change' : [0, 5, 10],
    'max_depth' : [2, 3, 4],
    
}
base_clf = GradientBoostingClassifier(criterion = 'mse', validation_fraction = 0.25, random_state = 0)

grid_search = RandomizedSearchCV(base_clf, params, scoring = 'accuracy', random_state = 0, n_iter = 40)
grid_search.fit(X_train[features], y_train)
print('40 random searches complete')

I evaluated the top three combinations from the randomized grid search using a learning curve.
# In[ ]:


grid_search_result = pd.DataFrame(grid_search.cv_results_)
grid_search_result.loc[grid_search_result['rank_test_score'] < 5].sort_values('rank_test_score')


# In[ ]:


clf_1 = GradientBoostingClassifier(n_estimators = 200,
                                 subsample = 0.45,
                                 max_features = 'auto',
                                 criterion = 'mse',
                                 n_iter_no_change = 10,
                                 learning_rate = 0.03,
                                 validation_fraction = 0.25,
                                 max_depth = 3, 
                                 random_state = 0)

clf_2 = GradientBoostingClassifier(n_estimators = 50,
                                 subsample = 0.7,
                                 max_features = 'sqrt',
                                 criterion = 'mse',
                                 n_iter_no_change = 10,
                                 learning_rate = 0.03,
                                 validation_fraction = 0.25,
                                 max_depth = 4, 
                                 random_state = 0)

clf_3 = GradientBoostingClassifier(n_estimators = 150,
                                 subsample = 0.7,
                                 max_features = 'auto',
                                 criterion = 'mse',
                                 n_iter_no_change = 5,
                                 learning_rate = 0.1,
                                 validation_fraction = 0.25,
                                 max_depth = 2, 
                                 random_state = 0)


# In[ ]:


from sklearn.model_selection import learning_curve

def plot_learning_curve(my_clf):
    n_samples, train_score, test_score = learning_curve(my_clf, X_train[features], y_train,
                                                        cv = 5, scoring = 'neg_mean_squared_error',
                                                        train_sizes = np.linspace(0.05, 1, 8),
                                                        random_state = 1)
    n_samples_mean = np.mean(n_samples)
    train_score_mean = -1 * np.mean(train_score, axis = 1)
    train_score_std = -1 * np.std(train_score, axis = 1)
    test_score_mean = -1 * np.mean(test_score, axis = 1)
    test_score_std =  -1 * np.std(test_score, axis = 1)
    
    plt.plot(n_samples, train_score_mean, '-o', color = 'darkorange', label = 'train')
    plt.fill_between(n_samples, 
                     train_score_mean + train_score_std, 
                     train_score_mean - train_score_std,
                     color = 'darkorange', alpha = 0.1)
    plt.plot(n_samples, test_score_mean,'-o', color = 'navy', label = 'test')
    plt.fill_between(n_samples, 
                     test_score_mean + test_score_std, 
                     test_score_mean - test_score_std,
                     color = 'navy', alpha = 0.1)
    plt.title('learning curve for gradient boosting model')
    plt.xlabel('train size')
    plt.ylabel('mse')
    plt.legend()
    plt.show()


# Learning curve notes:
# * The MSE is roughly 0.15 - 0.20 for all the curves
# * The second and third curve have larger splitting than the first curve, which suggests there is more overfitting in those models.

# In[ ]:


plot_learning_curve(clf_1)


# In[ ]:


plot_learning_curve(clf_2)


# In[ ]:


plot_learning_curve(clf_3)


# I did some manual, guess-and-check work to further optimize the hyperparameters and reduce overfitting by reducing the gap between the train and test curves.

# In[ ]:


final_clf = GradientBoostingClassifier(n_estimators = 150,
                                 subsample = 0.5,
                                 max_features = 'auto',
                                 criterion = 'mse',
                                 n_iter_no_change = 5,
                                 learning_rate = 0.07,
                                 validation_fraction = 0.25,
                                 max_depth = 3, 
                                 random_state = 0)
plot_learning_curve(final_clf)


# To confirm that the final model is accurate, I looked at the confusion matrix, ROC curve, and PR curve.

# In[ ]:


from sklearn.metrics import f1_score, accuracy_score

final_clf.fit(X_train[features], y_train)
preds = final_clf.predict(X_val[features])

print('Number of estimators after early stopping: {}'.format(final_clf.n_estimators_))
print('Accuracy: {}'.format(accuracy_score(preds, y_val)))
print('F1: {}'.format(f1_score(preds, y_val)))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(preds, y_val)
g = sns.heatmap(cm, annot = True, fmt = 'd')


# In[ ]:


from sklearn.metrics import plot_roc_curve

g = plot_roc_curve(final_clf, X_val[features], y_val)


# In[ ]:


from sklearn.metrics import plot_precision_recall_curve

g = plot_precision_recall_curve(final_clf, X_val[features], y_val)


# Preparing submission on test data

# In[ ]:


test = pd.read_csv('../input/titanic/test.csv')
X_test = test.copy()
X_test['Fare'] = np.log1p(X_test.Fare)
get_titles(X_test)


# In[ ]:


X_test_imp = pd.DataFrame(missing_transformer.transform(X_test))
X_test_imp.columns = numerical + categorical
X_test.drop(numerical + categorical, axis = 1, inplace = True)
X_test = pd.concat([X_test, X_test_imp], axis = 1)


# In[ ]:


X_test_enc = pd.DataFrame(ordinal_encoder.fit_transform(X_test[categorical]))
X_test_enc.columns = categorical
X_test.drop(categorical, axis = 1, inplace = True)
X_test = pd.concat([X_test, X_test_enc], axis = 1)
X_test = pd.get_dummies(X_test, columns = categorical, dtype = np.int64, drop_first = True)


# In[ ]:


X_test.head()


# In[ ]:


final_clf.fit(X[features], y)
final_preds = final_clf.predict(X_test[features])
final_submission = pd.DataFrame({'PassengerId' : X_test.PassengerId,
                               'Survived' : final_preds})
final_submission.to_csv('may12_finalpreds.csv', index = False)


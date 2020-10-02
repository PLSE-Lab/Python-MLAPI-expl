#!/usr/bin/env python
# coding: utf-8

# ![](https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg)

# # Titanic : three lines of code for LB = 0.79425

# Before making predictions with complex algorithms, I tried to make it simple. Early I developed a kernel (https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-20) that had three lines of code based on three statements and provides a LB of at least 80% of teams - Titanic Top 20%. Now I have figured out how to improve it. I added fourth statement and created 4 lines of code and then optimized this kernel again to 3 lines of main code.This provides a LB of at least 86% of teams - Titanic Top 14%, but the leaderboard is constantly changing - Titanic Top 15% will be more reliable. After the code I have justified in the form of graphs (EDA), from which the statements are obvious.

# Thanks to:
# 
# * [Three lines of code for Titanic Top 20%](https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-20)
# * [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# * [Titanic Top 3% : cluster analysis](https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis)
# * [Feature importance - xgb, lgbm, logreg, linreg](https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg)
# * [Tutorial 1: Gender Based Model (0.76555)](https://www.kaggle.com/mylesoneill/tutorial-1-gender-based-model-0-76555)
# * [Simplest Top 10% Titanic [0.80861]](https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861)
# * https://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb

# In[ ]:


import pandas as pd
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[ ]:


# Preparatory part of the code
test = pd.read_csv('../input/titanic/test.csv') # load test dataset
test['Boy'] = (test.Name.str.split().str[1] == 'Master.').astype('int')
test['Family'] = test['SibSp'] + test['Parch']
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pd.Series(dtype='int32')})

# Three lines of code for LB = 0.79425 (not less 85% teams - Titanic Top 15%) 
# Reasoning the statements see below (EDA)
# Statement 1
test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]
# Statement 2
test.loc[(test.Boy == 1), 'Survived'] = 1
# Statements 3,4
test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & ~((test.Boy == 1) & (test.Family > 0) & (test.Family < 4))), 'Survived'] = 0

# Saving the result
submission.Survived = test.Survived
submission.to_csv('submission_S_Boy_Sex_Family.csv', index=False)


# # Reasoning (EDA)

# ### Statement 1. Women all survived and men all died

# In[ ]:


# Reasoning for Statement 1 
# Thanks for the idea to: https://www.kaggle.com/mylesoneill/tutorial-1-gender-based-model-0-76555 
# Thanks for the idea of plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861
import matplotlib.pyplot as plt

def highlight(value):
    if value >= 0.5:
        style = 'background-color: palegreen'
    else:
        style = 'background-color: pink'
    return style

train = pd.read_csv('../input/titanic/train.csv') # load train dataset
pd.pivot_table(train, values='Survived', index=['Sex']).style.applymap(highlight)


# ### Statement 2. All boys ('Master') from the 1-2 classes survived

# In[ ]:


# Reasoning for Statement 2
# Thanks for the plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861
train['Boy'] = (train.Name.str.split().str[1] == 'Master.').astype('int')
pd.pivot_table(train, values='Survived', index='Pclass', columns='Boy').style.applymap(highlight)


# ### Statement 3. Everybody from the class 3 cabins that were sat in Southampton ('S') were died

# In[ ]:


# Reasoning for Statement 3
# Thanks for the plot to: https://www.kaggle.com/pavlofesenko/simplest-top-10-titanic-0-80861
pd.pivot_table(train, values='Survived', index=['Cabin', 'Embarked'], columns='Sex').style.applymap(highlight)


# ### Statement 4. The boys from the small families ('Family' = 1,2,3) of the third class cabins who were sitting in Southampton all survived

# In[ ]:


# Reasoning for Statement 4
train['Family'] = train['SibSp'] + train['Parch']
pd.pivot_table(train, values='Survived', index=['Pclass','Embarked','Boy','Family']).style.applymap(highlight)


# In[ ]:


# Statements 1,2,3,4 in 4 lines of code:
test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]
test.loc[(test.Boy == 1), 'Survived'] = 1
test.loc[((test.Pclass == 3) & (test.Embarked == 'S')), 'Survived'] = 0
test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & (test.Boy == 1) & (test.Family > 0) & (test.Family < 4)), 'Survived'] = 1


# In[ ]:


# Statements 1,2,3,4 in 3 lines of code (see above):
test['Survived'] = [1 if (x == 'female') else 0 for x in test['Sex']]
test.loc[(test.Boy == 1), 'Survived'] = 1
test.loc[((test.Pclass == 3) & (test.Embarked == 'S') & ~((test.Boy == 1) & (test.Family > 0) & (test.Family < 8))), 'Survived'] = 0


# In[ ]:


train 


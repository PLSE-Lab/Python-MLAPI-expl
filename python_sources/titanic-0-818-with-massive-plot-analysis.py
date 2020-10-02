#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


test_key = test.PassengerId


# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# # Analyze Missing values with sns heatmap

# In[ ]:


nan = dataset.isnull().sum()
plt.figure(figsize=(20,5))
idx_nan = nan.mask(nan==0).dropna().index
sns.heatmap(dataset[idx_nan].transpose().isnull(), cmap = 'binary')


# # Categorical features plot info
# Whit this custom funcion I can plot toghere more than 1 categorical feature with this informations:
# % of missing values for classes: (Survived, Dead)
# See % distribution of the categorial feature on dependent variable

# In[ ]:


def get_categorical_feature_info(data, column='', y_column='', positive_state='', negative_state='',                                  save_fig=False, path='', figsize=(10,10)):
    if column != y_column:
        total_positive = data[y_column].sum()
        total_negative = len(data)-total_positive
        positive = data[data[y_column]==1][column].value_counts()/total_positive
        negative = data[data[y_column]==0][column].value_counts()/total_negative
        df = pd.DataFrame([positive, negative])
        ind = (0,1)
        ax = df.plot(kind='bar',stacked=True, figsize=figsize)
        plt.xticks(ind, (positive_state, negative_state), fontsize=15)
        plt.yticks(np.arange(0, 1.1, step=0.1), fontsize=14)
        plt.ylim((0, 1.2))
        plt.ylabel('Value Count %', fontsize=15)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:.0%}'.format(x) for x in vals])
        plt.xlabel('Y value', fontsize=20)
        positive_missing_value = data[column][data[y_column]==1].isna().sum()/len(data[column][data[y_column]==1])
        negative_missing_value = data[column][data[y_column]==0].isna().sum()/len(data[column][data[y_column]==0])
        plt.title('Feature: '+str(column.upper())                  +'\nMissing values: '+str('{:,.2%}'.format(positive_missing_value))                  +positive_state+' and '+ str('{:,.2%}'.format(negative_missing_value)                                                  +negative_state), fontsize=15)
        if save_fig:
            file_name = path+str(column)+'.png'
            plt.savefig(file_name)


# In[ ]:


for dataset in train_test_data:
    dataset['Family_size'] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, 
                 "Master": 5, "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4,"Countess": 4,
                 "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona" : 4, "Mme": 4,"Capt": 4,"Sir": 4 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[ ]:


for dataset in train_test_data:
    dataset['Last_Name'] = dataset['Name'].apply(lambda x: str.split(x, ",")[0])


# In[ ]:


total_data = train.append(test, sort=False)


# # Missing cabin custom function
# With thi function I take cabin info for family and fill nan values

# In[ ]:


def fill_missing_cabin(data, match_data):
    cabin_list = []
    for i in range(len(data)):
        if data.Cabin[i] is np.nan:
            cabin_mode = match_data[(match_data['Last_Name'] == data['Last_Name'][i])                               & (data['Family_size'][i] == match_data['Family_size'])                               & (data['Pclass'][i] == match_data['Pclass'])].Cabin.mode()
            if len(cabin_mode) > 0:
                cabin_list.append(cabin_mode[0])
            else:
                cabin_list.append(np.nan)
        else:
            cabin_list.append(data.Cabin[i])
    return cabin_list


# In[ ]:


train['Cabin'] = fill_missing_cabin(train, total_data)
test['Cabin'] = fill_missing_cabin(test, total_data)


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin_letter'] = dataset['Cabin'].str[:1].fillna('Missing_cabin_info')


# In[ ]:


def get_cabin_count(data, column=''):
    cabin_count_list=[]
    for i in range(len(data)):
        if data[column][i] is np.nan:
            cabin_count_list.append(0)
        else:
            cabin_count_list.append(data[column][i].strip().count(' ')+1)
    return cabin_count_list


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin_count'] = get_cabin_count(dataset, column='Cabin')


# In[ ]:


sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[ ]:


categorical_features = ['Pclass', 'Sex', 'Cabin_letter', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp']


# In[ ]:


for column in categorical_features:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived',                                negative_state='Dead')


# # Filling missing values
# Fill nan Age with the median of same Title
# Fill nan Fare with media of same Pclass

# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# # Continuous features custom function
# With this function I can plot more the one continuous features using sns FacetGrid

# In[ ]:


def get_continuous_feature_info(data, column='', y_column='', xlim=None):
    facet = sns.FacetGrid(data, hue=y_column, aspect=4)
    facet.map(sns.kdeplot,column,shade= True)
    if xlim is None:
        facet.set(xlim=(0, data[column].max()))
    else:
        facet.set(xlim=xlim)
    facet.add_legend()
    plt.show()


# In[ ]:


continuous_features = [ 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Family_size']


# In[ ]:


for column in continuous_features:
    get_continuous_feature_info(train, column=column, y_column='Survived')


# # Filling missing Cabin info
# After first function to fill missing value in Cabin column, I saw that don't have a cabin info, is a "feature" to pass to
# train in the model because is distribution is different on the two classes of dependent variable

# In[ ]:


for dataset in train_test_data:
    dataset['Cabin_letter'] = dataset['Cabin_letter'].fillna('Missing_cabin_info')


# In[ ]:


for dataset in train_test_data:
    dataset['Cabin_count'] = dataset['Cabin_count'].fillna(0)


# In[ ]:


for column in ['Cabin_letter', 'Cabin_count']:
    get_categorical_feature_info(train, column=column, y_column='Survived')


# # Creating new features
# After analyzed "Missing cabin info", I decided to see if do not have a cabin is an important feature

# In[ ]:


for dataset in train_test_data:
    dataset['Has_a_Cabin'] = dataset["Cabin_letter"].apply(lambda x: 0 if x == 'Missing_cabin_info' else 1)
    dataset.drop(columns='Cabin_letter', inplace=True)


# In[ ]:


get_categorical_feature_info(train, column='Has_a_Cabin', y_column='Survived')


# In[ ]:


nan = dataset.isnull().sum()
plt.figure(figsize=(20,5))
idx_nan = nan.mask(nan==0).dropna().index
sns.heatmap(dataset[idx_nan].transpose().isnull(), cmap = 'binary')


# # Custom function to plot and extract outliers info
# With this function I can plot (with sns box plot) and extract outliers info with interquertile, lower boun and upper bound info

# In[ ]:


def get_outliers_info(data, column, figsize=(20,8)):
    plt.figure(figsize=figsize)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (IQR * 1.5)
    upper_bound = Q3 + (IQR * 1.5)
    bounds_list = [lower_bound, upper_bound]
    feature_bounds = bounds_list
    sns.boxplot(x=data[column])
    plt.title('IQR: '+str(IQR)+', Lower bound: '+str(lower_bound)+', Upper bound: '+str(upper_bound)+              '\nMedian: '+str(data[column].median()), fontsize=15)
    plt.show()
    return feature_bounds


# In[ ]:


outliers_bounds = {}
for column in continuous_features:
    outliers_bounds[column] = get_outliers_info(total_data, column=column)


# In[ ]:


outliers_bounds


# In[ ]:


for dataset in train_test_data:
    dataset.drop('Parch', axis='columns', inplace=True)


# In[ ]:


del outliers_bounds['Parch']
del outliers_bounds['Pclass']


# In[ ]:


outliers_bounds


# # Normalizing outliers
# With this funciton I normalized upper outliers (because in the plot there are not lower outliers) with upper bound value

# In[ ]:


def normalize_outliers(data, outliers_bounds={}):
    for key in outliers_bounds.keys():
        median = data[key].median()
        new_values = []
        for i in range(len(data)):
            value = data[key][i]
            if value < outliers_bounds[key][0] or value > outliers_bounds[key][1]:
                new_values.append(outliers_bounds[key][1])
            else:
                new_values.append(value)
        data[key] = new_values
    return data


# In[ ]:


train = normalize_outliers(train, outliers_bounds=outliers_bounds)
test = normalize_outliers(test, outliers_bounds=outliers_bounds)


# In[ ]:


total_data = train.append(test, sort=False)


# In[ ]:


del outliers_bounds['SibSp']


# In[ ]:


for column in outliers_bounds.keys():
    get_continuous_feature_info(train, column=column, y_column='Survived')


# # Custom binning for Fare
# Having info from the plot, I create custom bins for "Fare" variable.

# In[ ]:


for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 16), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 16) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 50), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 50, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)


# # Female features engineering
# 
# Now it's time to make questions.
# In first plots we saw that a woman has more probability to survive, but... <br>
# 1) What if she's the youngest woman in the family? <br>
# To answer to this and other questions, we have to match individual info with family info, so we have to have a DataFrame
# with all train and test info.

# In[ ]:


def is_youngest_girl(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Sex[i] == 1):
            match = (data.loc[i].Age <= min(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_Youngest_lady'] = is_youngest_girl(train, total_data)
test['Is_Youngest_lady'] = is_youngest_girl(test, total_data)


# 2) What if the passengers is a rich wife?

# In[ ]:


for dataset in train_test_data:
    dataset['Is_WifeandRich'] = ((dataset['Name'].str.find('(') > -1) & (dataset['Sex'] == 1) & (dataset['Pclass'] <3))*1


# In[ ]:


for column in ['Is_WifeandRich', 'Is_Youngest_lady']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# 3) What is she is a rich mother? (or a woman with children in her family?)

# In[ ]:


def is_a_rich_Mother(data, match_data):
    match_list = []
    for i in range(len(data)):
        if (data.Pclass[i] < 3) & (data.Sex[i] == 1):
            match = (14 >= min(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_a_rich_Mother'] = is_a_rich_Mother(train, total_data)
test['Is_a_rich_Mother'] = is_a_rich_Mother(test, total_data)


# 4) What if she is the only woman in the family?

# In[ ]:


def is_the_only_woman(data, match_data):
    match_list = []
    for i in range(len(data)):
        if data.Sex[i] == 1:
            match = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Sex.sum())
            if match == 1:
                match_list.append(match)
            else:
                 match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_the_only_woman'] = is_a_rich_Mother(train, total_data)
test['Is_the_only_woman'] = is_a_rich_Mother(test, total_data)


# In[ ]:


for column in ['Is_a_rich_Mother', 'Is_the_only_woman']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# 5) What if she has other women in her family?

# In[ ]:


def other_women_in_family(data, match_data):
    match_list = []
    for i in range(len(data)):
        if data.Sex[i] == 1:
            match = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Sex.sum())
            if match > 1:
                match_list.append(1)
            else:
                 match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Other_women_in_family'] = other_women_in_family(train, total_data)
test['Other_women_in_family'] = other_women_in_family(test, total_data)


# 6) What if she is the oldest woman in her family?

# In[ ]:


def is_oldest_woman(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Sex[i] == 1):
            match = (data.loc[i].Age == max(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_oldest_woman'] = is_oldest_woman(train, total_data)
test['Is_oldest_woman'] = is_oldest_woman(test, total_data)


# In[ ]:


for column in ['Is_oldest_woman', 'Other_women_in_family']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# # Male features engineering
# Now it's Male turn.
# 1) What if he is the youngest man in his family?

# In[ ]:


def is_youngest_boy(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Sex[i] == 0):
            match = (data.loc[i].Age <= min(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# 2) What if he is the oldest man in his family?

# In[ ]:


def is_oldest_man(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Sex[i] == 0):
            match = (data.loc[i].Age == max(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# 2) What if he has women in his family?

# In[ ]:


def man_with_woman(data, match_data):
    match_list = []
    for i in range(len(data)):
        if (data.Sex[i] == 0) & (data.Title[i] != 5):
            match = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])].Sex.sum())
            if match > 1:
                match_list.append(1)
            else:
                 match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# 2) What if he has other men in his family?

# In[ ]:


def man_with_other_men(data, match_data):
    match_list = []
    for i in range(len(data)):
        if data.Sex[i] == 0:
            match = len(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                   & (match_data['Sex'] == 0)].Sex)
            if match > 1:
                match_list.append(1)
            else:
                 match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_youngest_man'] = is_youngest_boy(train, total_data)
test['Is_youngest_man'] = is_youngest_boy(test, total_data)


# In[ ]:


train['Is_oldest_man'] = is_oldest_man(train, total_data)
test['Is_oldest_man'] = is_oldest_man(test, total_data)


# In[ ]:


train['Man_with_woman'] = man_with_woman(train, total_data)
test['Man_with_woman'] = man_with_woman(test, total_data)


# In[ ]:


train['Man_with_other_men'] = man_with_woman(train, total_data)
test['Man_with_other_men'] = man_with_woman(test, total_data)


# In[ ]:


for column in ['Is_youngest_man', 'Is_oldest_man', 'Man_with_woman', 'Man_with_other_men']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# # Children featuren engineering
# Now It's children turn.
# 1) is he/she the youngest one?

# In[ ]:


def is_youngest_children(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Title[i] == 5):
            match = (data.loc[i].Age <= min(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                   & (match_data['Title'] == 5)].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# 2) is he/she the oldest one?

# In[ ]:


def is_oldest_children(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Title[i] == 5):
            match = (data.loc[i].Age >= max(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                   & (match_data['Title'] == 5)].Age))*1

            match_list.append(match)
        else:
            match_list.append(0)
    return match_list


# 3) is he/she the only one?

# In[ ]:


def is_the_only_child(data, match_data):
    match_list = []
    for i in range(len(data)):
        if  (data.Title[i] == 5):
            match = len(match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                   & (match_data['Title'] == 5)])
            if match == 1:
                match_list.append(1)
            else:
                match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Is_youngest_children'] = is_youngest_children(train, total_data)
test['Is_youngest_children'] = is_youngest_children(test, total_data)


# In[ ]:


train['Is_oldest_children'] = is_oldest_children(train, total_data)
test['Is_oldest_children'] = is_oldest_children(test, total_data)


# In[ ]:


train['Is_the_only_child'] = is_the_only_child(train, total_data)
test['Is_the_only_child'] = is_the_only_child(test, total_data)


# In[ ]:


for column in ['Is_youngest_children', 'Is_oldest_children', 'Is_the_only_child']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# # Survived Family featuren engineering
# Now we have to analyze info about survived family, but we don't have to take passenger surviving info to do not have <br>
# overfitting result. <br>
# So, data['PassengerId'][i] != match_data['PassengerId']), exclude this case. <br>
# 1) Has survived family?

# In[ ]:


def family_survived(data, match_data):
    match_list = []
    for i in range(len(data)):
        if data.Family_size[i] > 1:
            survived = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                & (data['PassengerId'][i] != match_data['PassengerId'])].Survived).sum()
            if survived > 0:
                match_list.append(1)
            else:
                match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# 2) Is she a wife with survived children in her family?

# In[ ]:


def is_a_wife_with_survived_children(data, match_data):
    match_list = []
    for i in range(len(data)):
        if (data.Name[i].find('(') > -1) & (data.Sex[i] == 1):
            survived = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                & (data['PassengerId'][i] != match_data['PassengerId'])                                                & (match_data['Title'] == 5)].Survived).sum()
            if survived > 0:
                match_list.append(1)
            else:
                match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# Is He/She a child with other children or women survived?

# In[ ]:


def is_a_child_with_other_candw_survived(data, match_data):
    match_list = []
    for i in range(len(data)):
        if (data.Title[i] == 5):
            survived = (match_data[(match_data['Last_Name'] == data['Last_Name'][i])                                                   & (data['Family_size'][i] == match_data['Family_size'])                                                    & (data['Pclass'][i] == match_data['Pclass'])                                                & (data['PassengerId'][i] != match_data['PassengerId'])                                                & ((match_data['Title'] == 5) | (match_data['Sex'] == 0))].Survived).sum()
            if survived > 0:
                match_list.append(1)
            else:
                match_list.append(0)
        else:
            match_list.append(0)
    return match_list


# In[ ]:


train['Family_survived'] = family_survived(train, total_data)
test['Family_survived'] = family_survived(test, total_data)


# In[ ]:


train['Is_a_wife_with_sc'] = is_a_wife_with_survived_children(train, total_data)
test['Is_a_wife_with_sc'] = is_a_wife_with_survived_children(test, total_data)


# In[ ]:


train['Is_a_child_with_wandc_survived'] = is_a_child_with_other_candw_survived(train, total_data)
test['Is_a_child_with_wandc_survived'] = is_a_child_with_other_candw_survived(test, total_data)


# In[ ]:


for column in ['Family_survived', 'Is_a_wife_with_sc', 'Is_a_child_with_wandc_survived']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# In[ ]:


for value in [(0,15), (15,35), (35,50), (50,65), (65,100)]:
    get_continuous_feature_info(train, column='Age', y_column='Survived', xlim=value)


# Now, with Age plot, I can bin Age column

# In[ ]:


for dataset in train_test_data:    
    dataset.loc[ dataset['Age'] <= 14, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


# In[ ]:


test.isna().sum(), train.isna().sum()


# In[ ]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# # Filling Embarked missing values with the most frequent info

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[ ]:


embarked_mapping = {"C": "Cherbourg", "S": "Southampton", "Q": "Queenstown"}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # Class and status features engineering

# In[ ]:


for dataset in train_test_data:
    dataset['Is_AloneandPoor'] = 0
    dataset.loc[(dataset['Family_size'] == 1) & (dataset['Pclass'] == 3 ),'Is_AloneandPoor'] = 1


# In[ ]:


get_categorical_feature_info(train, column='Is_AloneandPoor', y_column='Survived', positive_state='Survived', negative_state='Dead')


# In[ ]:


for dataset in train_test_data:
    dataset['IsChildandRich'] = 0
    dataset.loc[(dataset['Title'] == 5) & (dataset['Pclass'] < 3 ),'IsChildandRich'] = 1  


# In[ ]:


get_categorical_feature_info(train, column='IsChildandRich', y_column='Survived', positive_state='Survived', negative_state='Dead')


# In[ ]:


for data in train_test_data:
    data['Cabin'] = data['Cabin'].fillna('X')
    data['Cabin'] = data['Cabin'].apply(lambda x: str(x)[0])
    data['Cabin'] = data['Cabin'].replace(['A', 'D', 'E', 'T'], 'M')
    data['Cabin'] = data['Cabin'].replace(['B', 'C'], 'H')
    data['Cabin'] = data['Cabin'].replace(['F', 'G'], 'L')


# In[ ]:


get_categorical_feature_info(train, column='Cabin', y_column='Survived', positive_state='Survived', negative_state='Dead')


# In[ ]:


for dataset in train_test_data:
    dataset['Family_size_group'] = 'Small'
    dataset.loc[dataset['Family_size'] == 1, 'Family_size_group'] = 'Alone'
    dataset.loc[dataset['Family_size'] > 2, 'Family_size_group'] = 'Big'


# In[ ]:


train['IsMaleandPoor'] = 0
train.loc[(train['Sex'] == 0) & (train['Pclass'] == 3 ),'IsMaleandPoor'] = 1
test['IsMaleandPoor'] = 0
test.loc[(test['Sex'] == 0) & (test['Pclass'] == 3 ),'IsMaleandPoor'] = 1 


# In[ ]:


for column in ['Family_size_group', 'IsMaleandPoor']:
    get_categorical_feature_info(train, column=column, y_column='Survived', positive_state='Survived', negative_state='Dead')


# In[ ]:


train.columns


# # Dropping useless columns

# In[ ]:


for dataset in train_test_data:
    dataset.drop(['PassengerId', 'Name', 'Ticket', 'Last_Name', 'SibSp', 'Ticket'                 , 'Family_size'], axis=1, inplace=True)


# In[ ]:


len(train.columns), len(test.columns)


# In[ ]:


train.dtypes


# In[ ]:


def get_dummies_from_list(categorical_list):
    dummies = pd.get_dummies(pd.Categorical(categorical_list), sparse=True)
    return dummies


# In[ ]:


def transform_dummies(data, column_list=[]):
    for column in column_list:
        dummy_df = get_dummies_from_list(data[column])
        data = data.merge(dummy_df, how='left', left_index=True, right_index=True)
    data.drop(labels=column_list, axis='columns', inplace=True)
    return data


# # Creationg dummy columns for categorical variables

# In[ ]:


train = transform_dummies(train, column_list=['Embarked', 'Family_size_group', 'Cabin'])
test = transform_dummies(test, column_list=['Embarked', 'Family_size_group', 'Cabin'])


# In[ ]:


len(train.columns), len(test.columns)


# # Custom function to plot info about 2 indipendent variables on dependent variable

# In[ ]:


def get_indipendent_variables_mix_info(data, x_column='', y_column='', dependent_variable=''):
    if (x_column != y_column) & (x_column != dependent_variable):
        fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,14))
        sns.boxplot(x = x_column, y = y_column, hue = dependent_variable, data = data, ax = axis1)
        axis1.set_title(str(x_column)+' vs '+str(y_column))
        sns.violinplot(x = x_column, y = y_column, hue = dependent_variable, data = data, ax = axis2, split = True)
        axis2.set_title(str(x_column)+' vs '+str(y_column))
    plt.show()


# In[ ]:


for column in train.columns:
    get_indipendent_variables_mix_info(train, x_column=column, y_column='Age', dependent_variable='Survived')


# In[ ]:


corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


abs(corr.Survived).sort_values(ascending=False)


# # PCA Analysis on 2 dimensions

# In[ ]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import axes3d

plt.figure(figsize=(10,10))
train2D = PCA(n_components=2).fit_transform(train.drop('Survived', axis='columns'))
plt.scatter(train2D[train[train.Survived==0].reset_index(drop=True).index, 0], train2D[train[train.Survived==0].reset_index(drop=True).index, 1], c='red', marker='.', alpha=.6, )
plt.scatter(train2D[train[train.Survived==1].reset_index(drop=True).index, 0], train2D[train[train.Survived==1].reset_index(drop=True).index, 1], c='green', marker='.', alpha=.2)

plt.show()


# # PCA Analysis on 3 dimensions

# In[ ]:


ax = Axes3D(plt.figure(figsize=(10,10)))
train3D = PCA(n_components=3).fit_transform(train.drop('Survived', axis='columns'))
ax.scatter3D(train3D[train[train.Survived==0].reset_index(drop=True).index, 0], train3D[train[train.Survived==0].reset_index(drop=True).index, 1], train3D[train[train.Survived==0].reset_index(drop=True).index, 2], color='red', marker='.', alpha=.25, s=450)
ax.scatter3D(train3D[train[train.Survived==1].reset_index(drop=True).index, 0], train3D[train[train.Survived==1].reset_index(drop=True).index, 1], train3D[train[train.Survived==1].reset_index(drop=True).index, 2], color='green', marker='.', alpha=.9, s=300)
ax.view_init(elev=25, azim=45)

plt.show()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np


# In[ ]:


train_data = train.drop(columns='Survived')
target = train['Survived']

train_data.shape, target.shape


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, make_scorer, recall_score, precision_score, average_precision_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train.Survived, test_size=0.10, stratify=train['Survived'].values, random_state=42)


# # CrossValidation on 10 splits to test my model

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[ ]:


clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', probability=True)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
print('The total score is: '+str(score.mean()))


# In[ ]:


clf.fit(train_data, target)


# In[ ]:


predictions = clf.predict(X_test)


# In[ ]:


y_score = clf.predict_proba(X_test)


# In[ ]:


average_precision = average_precision_score(y_test, y_score[:, 0])
average_precision


# In[ ]:


import scikitplot as skplt


# In[ ]:


skplt.metrics.plot_precision_recall(y_test, y_score, figsize=(15,10))
plt.show()


# In[ ]:


skplt.metrics.plot_roc(y_test, y_score, figsize=(15,10))
plt.show()


# In[ ]:


test_prediction = clf.predict(test)


# In[ ]:


prediction = clf.predict(test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_key,
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# In[ ]:





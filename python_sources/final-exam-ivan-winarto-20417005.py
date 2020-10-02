#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


accidents = pd.read_csv('/kaggle/input/uk-road-safety-accidents-and-vehicles/Accident_Information.csv')
print('Records:', accidents.shape[0], '\nColumns:', accidents.shape[1])
accidents.head()


# In[ ]:


vehicles = pd.read_csv('/kaggle/input/uk-road-safety-accidents-and-vehicles/Vehicle_Information.csv', encoding='ISO-8859-1')
print('Records:', vehicles.shape[0], '\nColumns:', vehicles.shape[1])
vehicles.head()


# In[ ]:


accidents['Date']= pd.to_datetime(accidents['Date'], format="%Y-%m-%d")


# In[ ]:


accidents.iloc[:, 5:13].info()


# In[ ]:



sns.set_style('white')
fig, ax = plt.subplots(figsize=(15,6))


accidents.set_index('Date').resample('M').size().plot(label='total per bulan', color='green', ax=ax)
accidents.set_index('Date').resample('M').size().rolling(window=10).mean()                           .plot(color='yellow', linewidth=5, label='Rata- rata dalam dala 10 bulan', ax=ax)

ax.set_title('Kecelakaan dalam sebulan', fontsize=14, fontweight='bold')
ax.set(ylabel='Jumlah\n', xlabel='')
ax.legend(bbox_to_anchor=(1.1, 1.1), frameon=False)


sns.despine(ax=ax, top=True, right=True, left=True, bottom=False);


# In[ ]:



accidents['Hour'] = accidents['Time'].str[0:2]


accidents['Hour'] = pd.to_numeric(accidents['Hour'])


accidents = accidents.dropna(subset=['Hour'])


accidents['Hour'] = accidents['Hour'].astype('int')


# In[ ]:



def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "morning rush (5-10)"
    elif hour >= 10 and hour < 15:
        return "office hours (10-15)"
    elif hour >= 15 and hour < 19:
        return "afternoon rush (15-19)"
    elif hour >= 19 and hour < 23:
        return "evening (19-23)"
    else:
        return "night (23-5)"


# In[ ]:



accidents['Daytime'] = accidents['Hour'].apply(when_was_it)
accidents[['Time', 'Hour', 'Daytime']].head(30)


# In[ ]:


yearly_count = accidents['Date'].dt.year.value_counts().sort_index(ascending=False)

# prepare plot
sns.set_style('white')
fig, ax = plt.subplots(figsize=(12,5))

# plot
ax.bar(yearly_count.index, yearly_count.values, color='red')
ax.plot(yearly_count, linestyle=':', color='black')
ax.set_title('\nKecelakaan per tahun\n', fontsize=14, fontweight='bold')
ax.set(ylabel='\nJumlah')

# remove all spines
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);


# In[ ]:


weekday_counts = pd.DataFrame(accidents.set_index('Date').resample('1d')['Accident_Index'].size().reset_index())
weekday_counts.columns = ['Date', 'Count']


weekday = weekday_counts['Date'].dt.weekday_name


weekday_averages = pd.DataFrame(weekday_counts.groupby(weekday)['Count'].mean().reset_index())
weekday_averages.columns = ['Weekday', 'Average_Accidents']
weekday_averages.set_index('Weekday', inplace=True)


# In[ ]:



days = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']


sns.set_style('white')
fig, ax = plt.subplots(figsize=(10,5))
colors=['green', 'red', 'grey', 'pink', 
        'brown', 'orange', 'lightsteelblue']


weekday_averages.reindex(days).plot(kind='barh', ax=ax, color=[colors])
ax.set_title('\nRata - rata kecelakaan di weekday\n', fontsize=14, fontweight='bold')
ax.set(xlabel='\nRata - rata jumlah', ylabel='')
ax.legend('')


sns.despine(ax=ax, top=True, right=True, left=True, bottom=True);


# In[ ]:


accidents.Weather_Conditions.value_counts(normalize=True)


# In[ ]:


accidents.Accident_Severity.value_counts()


# In[ ]:



fatal   = accidents.Accident_Severity.value_counts()['Fatal']
serious = accidents.Accident_Severity.value_counts()['Serious']
slight  = accidents.Accident_Severity.value_counts()['Slight']

names = ['Kecelakaan parah','Kecelakaan sedang', 'Kecelakaan ringan']
size  = [fatal, serious, slight]
#explode = (0.2, 0, 0)


plt.pie(x=size, labels=names, colors=['red', 'yellow', 'green'], 
        autopct='%1.2f%%', pctdistance=0.6, textprops=dict(fontweight='bold'),
        wedgeprops={'linewidth':7, 'edgecolor':'white'})


my_circle = plt.Circle((0,0), 0.6, color='white')


fig = plt.gcf()
fig.set_size_inches(8,8)
fig.gca().add_artist(my_circle)
plt.title('\nTingkat kecelakaan di tahun 2013-2017', fontsize=14, fontweight='bold')
plt.show()


# In[ ]:


sub_df = accidents[['Date', 'Accident_Index', 'Accident_Severity']]


year = sub_df['Date'].dt.year
week = sub_df['Date'].dt.week

count_of_fatalities = sub_df.set_index('Date').groupby([pd.Grouper(freq='W'), 'Accident_Severity']).size()

fatalities_table = count_of_fatalities.rename_axis(['Week', 'Accident_Severity'])                                      .unstack('Accident_Severity')                                      .rename({1:'fatal', 2:'serious', 3:'slight'}, axis='columns')


# In[ ]:


fatalities_table['sum'] = fatalities_table.sum(axis=1)
fatalities_table = fatalities_table.join(fatalities_table.div(fatalities_table['sum'], axis=0), rsuffix='_percentage')


# In[ ]:



sub_df = fatalities_table[['Fatal_percentage', 'Serious_percentage', 'Slight_percentage']]

sns.set_style('white')
fig, ax = plt.subplots(figsize=(14,6))
colors=['green', 'pink', 'red']

sub_df.plot(color=colors, ax=ax)
ax.set_title('\nProporsi Tingkat Keparahan Kecelakaan\n', fontsize=14, fontweight='bold')
ax.set(ylabel='\n', xlabel='')
ax.legend(labels=['Kecelakaan Parah', 'Kecelakaan Sedang', 'Kecelakaan Ringan'], 
          bbox_to_anchor=(1.3, 1.1), frameon=False)

sns.despine(top=True, right=True, left=True, bottom=False);


# In[ ]:



sns.set_style('white')
fig, ax = plt.subplots(figsize=(10,6))


accidents.Hour.hist(bins=24, ax=ax, color='brown')
ax.set_title('\nKecelakaan di jam\n', fontsize=14, fontweight='bold')
ax.set(xlabel='jam', ylabel='Jumlah kecelakaan')


sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


order = ['night (23-5)', 'evening (19-23)', 'afternoon rush (15-19)', 'office hours (10-15)', 'morning rush (5-10)']
df_sub = accidents.groupby('Daytime').size().reindex(order)


fig, ax = plt.subplots(figsize=(10, 5))
colors = ['green', 'green', 'orange', 'green', 'green']


df_sub.plot(kind='barh', ax=ax, color=colors)
ax.set_title('\nKecelakaan pada waktu\n', fontsize=14, fontweight='bold')
ax.set(xlabel='\nTjumlah kecelakaan', ylabel='')


sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:



counts = accidents.groupby(['Daytime', 'Accident_Severity']).size()

counts = counts.rename_axis(['Daytime', 'Accident_Severity'])                                .unstack('Accident_Severity')                                .rename({1:'fatal', 2:'serious', 3:'slight'}, axis='columns')


# In[ ]:


counts['sum'] = counts.sum(axis=1)
counts = counts.join(counts.div(counts['sum'], axis=0), rsuffix=' in %')
counts_share = counts.drop(columns=['Fatal', 'Serious', 'Slight', 'sum', 'sum in %'], axis=1)


# In[ ]:


# prepare barplot
fig, ax = plt.subplots(figsize=(10, 5))

# plot
counts_share.reindex(order).plot(kind='barh', ax=ax, stacked=True, cmap='cividis')
ax.set_title('\nkecelakaan pada waktu\n', fontsize=14, fontweight='bold')
ax.set(xlabel='persentase', ylabel='')
ax.legend(bbox_to_anchor=(1.25, 0.98), frameon=False)

# remove all spines
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


vehicles.Sex_of_Driver.value_counts(normalize=True)


# In[ ]:


drivers = vehicles.groupby(['Age_Band_of_Driver', 'Sex_of_Driver']).size().reset_index()

# drop the values that have no value
drivers.drop(drivers[(drivers['Age_Band_of_Driver'] == 'Data missing or out of range') |                      (drivers['Sex_of_Driver'] == 'Not known') |                      (drivers['Sex_of_Driver'] == 'Data missing or out of range')]                     .index, axis=0, inplace=True)
# rename the columns
drivers.columns = ['Age_Band_of_Driver', 'Sex_of_Driver', 'Count']
drivers


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 7))
sns.barplot(y='Age_Band_of_Driver', x='Count', hue='Sex_of_Driver', data=drivers, palette='bone')
ax.set_title('\nPerbandingan kecelakaan dengan umur dan jenis kelamin\n', fontsize=14, fontweight='bold')
ax.set(xlabel='Count', ylabel='Age Band of Driver')
ax.legend(bbox_to_anchor=(1.1, 1.), borderaxespad=0., frameon=False)

sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


accidents['Date']= pd.to_datetime(accidents['Date'], format="%Y-%m-%d")


# In[ ]:


accidents.iloc[:, 8:11].info()


# In[ ]:


daytime_groups = {1: 'Morning (5-10)', 
                  2: 'Office Hours (10-15)', 
                  3: 'Afternoon Rush (15-19)', 
                  4: 'Evening (19-23)', 
                  5: 'Night(23-5)'}


# In[ ]:


# slice first and second string from time column
accidents['Hour'] = accidents['Time'].str[0:2]

# convert new column to numeric datetype
accidents['Hour'] = pd.to_numeric(accidents['Hour'])

# drop null values in our new column
accidents = accidents.dropna(subset=['Hour'])

# cast to integer values
accidents['Hour'] = accidents['Hour'].astype('int')


# In[ ]:


# define a function that turns the hours into daytime groups
def when_was_it(hour):
    if hour >= 5 and hour < 10:
        return "1"
    elif hour >= 10 and hour < 15:
        return "2"
    elif hour >= 15 and hour < 19:
        return "3"
    elif hour >= 19 and hour < 23:
        return "4"
    else:
        return "5"
    
# apply this function to our temporary hour column
accidents['Daytime'] = accidents['Hour'].apply(when_was_it)
accidents[['Time', 'Hour', 'Daytime']].tail()


# In[ ]:


accidents = accidents.drop(columns=['Time', 'Hour'])


# In[ ]:


labels = tuple(daytime_groups.values())

# plot total no. of accidents by daytime
accidents.groupby('Daytime').size().plot(kind='bar', color='green', figsize=(12,5), grid=True)
plt.xticks(np.arange(5), labels, rotation='horizontal')
plt.xlabel(''), plt.ylabel('Jumlah\n')
plt.title('\nTotal jumlah kecelakaan dari jam\n', fontweight='bold')
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


accidents.groupby('Daytime')['Number_of_Casualties'].mean().plot(kind='bar', color='orange', 
                                                                 figsize=(12,4), grid=False)
plt.xticks(np.arange(5), labels, rotation='horizontal')
plt.ylim((1,1.5))
plt.xlabel(''), plt.ylabel('rata rata corbans\n')
plt.title('\nJumlah rata - rata korban di lihat dari jam\n', fontweight='bold')
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


print('Proportion of Missing Values in Accidents Table:', 
      round(accidents.isna().sum().sum()/len(accidents),3), '%')


# In[ ]:


accidents = accidents.drop(columns=['Location_Easting_OSGR', 'Location_Northing_OSGR', 
                                    'Longitude', 'Latitude'])

# drop remaining records with NaN's
accidents = accidents.dropna()

# check if we have no NaN's anymore
accidents.isna().sum().sum()


# In[ ]:


df = accidents[['Accident_Index', 'Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week', 
                'Daytime', 'Road_Type', 'Speed_limit', 'Urban_or_Rural_Area', 'LSOA_of_Accident_Location']]
df.isna().sum().sum()


# In[ ]:


for col in ['Accident_Severity', 'Day_of_Week', 'Daytime', 'Road_Type', 'Speed_limit', 
            'Urban_or_Rural_Area', 'LSOA_of_Accident_Location']:
    df[col] = df[col].astype('category')
    
df.info()


# In[ ]:


df.groupby('Road_Type')['Number_of_Casualties'].mean().plot(kind='bar', color='pink', 
                                                            figsize=(12,4), grid=False)
plt.xticks(np.arange(6), 
           ['Roundabout', 'One way street', 'Dual carriageway', 'Single carriageway', 'Slip road', 'Unknown'], 
           rotation='horizontal')
plt.ylim((1,1.5))
plt.xlabel(''), plt.ylabel('angka rata rata\n')
plt.title('\nRata - rata kecelakaan di karena jenis jalan\n', fontweight='bold')
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


df.groupby('Speed_limit')['Number_of_Casualties'].mean().plot(kind='bar', color='lightgreen', 
                                                              figsize=(15,4), grid=False)
plt.xticks(np.arange(8), 
           ['None', '10mph', '20mph', '30mph', '40mph', '50mph', '60mph', '70mph'], 
           rotation='horizontal')
plt.ylim((0.6,1.6))
plt.xlabel(''), plt.ylabel('Angka rata-rata\n')
plt.title('\nRata - rata angka kecelakaan karena batas kecelakaan\n', fontweight='bold')
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


df.groupby('Day_of_Week')['Number_of_Casualties'].mean().plot(kind='bar', color='brown', 
                                                              figsize=(14,4), grid=False)
plt.xticks(np.arange(7), 
           ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], 
           rotation='horizontal')
plt.ylim((1.0,1.6))
plt.xlabel(''), plt.ylabel('Angka Rat-Rata\n')
plt.title('\nAngka Rata-rata kecelakaan di hari kerja\n', fontweight='bold')
sns.despine(top=True, right=True, left=True, bottom=True);


# In[ ]:


num_cols = ['Number_of_Vehicles', 'Number_of_Casualties']


# In[ ]:


sns.set(style='darkgrid')
fig, axes = plt.subplots(2,1, figsize=(10,4))

for ax, col in zip(axes, num_cols):
    df.boxplot(column=col, grid=False, vert=False, ax=ax)
    plt.tight_layout();


# In[ ]:


df['Number_of_Vehicles'].value_counts().head(10)


# In[ ]:


df['Number_of_Casualties'].value_counts().head(20)


# In[ ]:


condition = (df['Number_of_Vehicles'] < 6) & (df['Number_of_Casualties'] < 9)

df = df[condition]


print(df['Number_of_Vehicles'].value_counts())


# In[ ]:


print(df['Number_of_Casualties'].value_counts())


# In[ ]:


df.head(2)


# In[ ]:


look_up = pd.read_csv('../input/datasetex/Output_Area_to_LSOA_to_MSOA_to_Local_Authority_District__December_2017__Lookup_with_Area_Classifications_in_Great_Britain.csv')
look_up.head(10)


# In[ ]:


df_merged = pd.merge(df, look_up[['LSOA11CD', 'LAD17NM']], how='left', 
                     left_on='LSOA_of_Accident_Location', right_on='LSOA11CD')
df_merged.head(5)


# In[ ]:


df_merged = df_merged.drop(columns=['LSOA_of_Accident_Location', 'LSOA11CD'])                        .rename(columns={'LAD17NM': 'County_of_Accident'})                            .astype({'County_of_Accident': 'category'})                                .drop_duplicates()

df_merged.head(5)


# In[ ]:


df_merged.shape


# In[ ]:


df_merged.groupby('County_of_Accident').size().sort_values(ascending=False).head()


# In[ ]:


df_plot = df_merged.groupby('County_of_Accident').size().reset_index().rename(columns={0:'Count'})
df_plot.head()


# In[ ]:


num_col = ['Number_of_Vehicles']

cat_cols = ['Accident_Severity', 'Day_of_Week', 'Daytime', 'Road_Type', 'Speed_limit', 
            'Urban_or_Rural_Area', 'County_of_Accident']

target_col = ['Number_of_Casualties']

cols = cat_cols + num_cols + target_col

df_model = df_merged[cols].copy()
df_model.shape


# In[ ]:


dummies = pd.get_dummies(df_model[cat_cols], drop_first=True)
df_model = pd.concat([df_model[num_cols], df_model[target_col], dummies], axis=1)
df_model.shape


# In[ ]:


df_model.isna().sum().sum()


# ### Train-Test-Split

# In[ ]:


features = df_model.drop(['Number_of_Casualties'], axis=1)

target = df_model[['Number_of_Casualties']]


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)


# In[ ]:


# import regressor
from sklearn.ensemble import RandomForestRegressor

# import metrics
from sklearn.metrics import mean_squared_error, r2_score

# import evaluation tools
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


forest = RandomForestRegressor(random_state=4, n_jobs=-1)

# train
forest.fit(X_train, y_train)

# predict
y_train_preds = forest.predict(X_train)
y_test_preds  = forest.predict(X_test)

# evaluate
RMSE = np.sqrt(mean_squared_error(y_test, y_test_preds))
print(f"RMSE: {round(RMSE, 4)}")

r2 = r2_score(y_test, y_test_preds)
print(f"r2: {round(r2, 4)}")


# In[ ]:


print('Parameters currently in use:\n')
pprint(forest.get_params())


# In[ ]:


n_estimators = [100, 150]

max_depth = [3, 4, 5]

min_samples_split = [10, 15, 20]

hyperparameters = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
hyperparameters


# In[ ]:


randomized_search = RandomizedSearchCV(forest, hyperparameters, n_jobs=-1)

best_model = randomized_search.fit(X_train, y_train)

print(best_model.best_params_)


# In[ ]:


forest = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=4, n_jobs=-1)

forest.fit(X_train, y_train)

y_train_preds = forest.predict(X_train)
y_test_preds  = forest.predict(X_test)

RMSE = np.sqrt(mean_squared_error(y_test, y_test_preds))
print(f"RMSE: {round(RMSE, 4)}")

r2 = r2_score(y_test, y_test_preds)
print(f"r2: {round(r2, 4)}")


# In[ ]:


feat_importances = pd.Series(forest.feature_importances_, index=features.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))
plt.xlabel('Relative Feature Importance with Random Forest');


# In[ ]:


from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_log = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log


# In[ ]:


svc = SVC()
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred_gaussian = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred_perceptron = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred_linear_svc = linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred_sgd = sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd


# In[ ]:


tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
acc_tree = round(tree.score(x_train, y_train) * 100, 2)
acc_tree


# In[ ]:


forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)
y_pred_forest = forest.predict(x_test)
acc_forest = round(forest.score(x_train, y_train) * 100, 2)
acc_forest


# In[ ]:


models = pd.DataFrame({
    'Model':['Logistic Regression' , 'SVM' , 'KNN' , 'Gaussian' , 'Perceptron' , 'Linear SVC' , 'SGD' , 'Tree' , 'Forest'],
    'Score':[acc_log , acc_svc , acc_knn , acc_gaussian , acc_perceptron , acc_linear_svc , acc_sgd , acc_tree , acc_forest]
})
models.sort_values(by='Score' , ascending=False)


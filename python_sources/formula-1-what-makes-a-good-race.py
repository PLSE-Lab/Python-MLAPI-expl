#!/usr/bin/env python
# coding: utf-8

# # Formula 1: What makes a good race?
# All racing fans can identify a "good race" when they see one, but exactly what is it that we see in the race that makes us rate it as "good" or "bad"? Using some data science, we will look to see if we can find the "key" to a good F1 race!

# We start by importing all python libraries that we will need to perform our analysis

# In[ ]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


# The data we will use in our analysis is the full Formula 1 dataset, spanning all drivers, races, circuits, teams etc. from the 1950's to 2018. The data which we will use was generously uploaded by user Chris G (cjgdev), and you can find all the files, including their description here https://www.kaggle.com/cjgdev/formula-1-race-data-19502017. The full dataset includes several .csv-files, but we will focus on a few specifc ones.

# In[ ]:


folder_path = '/kaggle/input/formula-1-race-data-19502017/'
lapTimes = pd.read_csv(folder_path + 'lapTimes.csv', encoding='latin-1')
races = pd.read_csv(folder_path + 'races.csv', encoding='latin-1')
drivers = pd.read_csv(folder_path + 'drivers.csv', encoding='latin-1')
results = pd.read_csv(folder_path + 'results.csv', encoding='latin-1')
circuits = pd.read_csv(folder_path + 'circuits.csv', encoding='latin-1')
status = pd.read_csv(folder_path + 'status.csv', encoding='latin-1')


# I have gathered data from https://www.racefans.net on both their "Top 100 F1 races" list, as well as the complete seasons 2014, 2015 and 2016. We load them now and I'll start by exploring a baseline of scores based on circuit and year.

# In[ ]:


folder_path_2 = '/kaggle/input/formula1addonscores/'
top_100 = pd.read_csv(folder_path_2 + 'top_100.csv', encoding='latin-1', index_col = False)
top_100 = top_100.drop(['Unnamed: 0'], axis=1)
scores = pd.read_excel(folder_path_2 + 'score_season_2014_2015_2016.xlsx', header=0)
print(top_100.columns)
print(scores.columns)
print(top_100.head())
print(scores.head())


# In[ ]:


fig = plt.figure(figsize=(27,9))
plt.subplot(1,3,1)
plt.title('Scatter plot of race scores vs year')
_ = sns.scatterplot(x='year', y='points', data=top_100)
plt.xticks(rotation=45)
plt.subplot(1,3,2)
plt.title('Box plot of the race score distribution per year')
_ = sns.boxplot(x='year', y='points', data=top_100)
plt.xticks(rotation='45')
plt.subplot(1,3,3)
plt.title('The amount of races our score dataset contains per year')
_ = sns.countplot(x='year', data=top_100)
plt.xticks(rotation='45')
plt.show()


# For further analysis, we will merge the top_100 score with the full season data of 2014, 2015 and 2016 to get one dataframe containing raceId and points

# In[ ]:


df_scores = pd.concat([top_100, scores])
df_scores = df_scores.drop_duplicates(subset='raceId')
print(df_scores.head())


# A first thought might be that certain tracks are more favoured than others and will get a higher score regardless of how the race turned out. Due to the limited amount of data, it might be difficult to do a full analysis on how the track affects the score, but a quick look at the top 10 races vs bottom 10 races might share some insight if a track appears more often in any of those categories.

# In[ ]:


df_scores.sort_values('points', ascending=False).head(10).name.value_counts()


# In[ ]:


df_scores.sort_values('points', ascending=False).tail(10).name.value_counts()


# In[ ]:


print(df_scores.groupby('name').points.mean().sort_values(ascending=False))


# We can see that in our dataset, the Russian and Mexican grand prix have significantly lower average score, with Japanese and Singapore grand prix have slightly worse than the average. Four grand prix score higher than 7.9, while the vast majority lies between 7.0 and 7.9.

# # Hypothesis 1: Amount of DNFs affects the score
# The first hypothesis is that the amount of drivers not finishing a race is correlated with the race score. Some might argue that "more DNFs = more action", while other might think that a thinning race lineup will make it less interesting to watch to the end. Looking at the results dataframe, we can study the Barcelona Grand Prix 2017, which is raceId = 973 and see how the dataset handles DNFs.

# In[ ]:


print(results[results['raceId'] == 973][['resultId', 'raceId', 'driverId','grid', 'position', 'statusId']])


# We see that drivers 822, 838, 830 and 8 did not get a position in the race, due to status 131, 4 and 130. Looking in the status dataframe, we can see what this mean:

# In[ ]:


print(status[status['statusId'].isin([131, 4, 130])])


# These cars did not finish due to collisions and a failed power unit.

# We create our analysis dataframe, df, by performing a left join on our score dataframe with the races dataframe, and we see that we have 120 races for our analysis.

# In[ ]:


df = pd.merge(left=df_scores, right=races, on='raceId', how='left')
df = df.dropna(axis=0)
dropIndex = df[df['raceId'] > 988].index
df.drop(dropIndex, inplace=True)
print(df.columns)
print(len(df))


# To count the number of DNF in a race, we create a function which takes the results dataframe and a race-id as input. It then filters the result dataframe based on raceid, and goes through each result to see if the status is in a given list of "finished" statuses. 

# In[ ]:


print(status[status['statusId'].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19])])


# In[ ]:


def count_dnf(_results, _raceid):
    # Count the number of "did not finished" per race
    # for each result, there is a statusId and a collection of these signify that the driver
    # finished,
    dnf_count = 0
    finished_status = [1, 11, 12, 13, 14, 15, 16, 17, 18, 19]   # theses statusId are given when a driver finishes

    for iter_status in _results[_results['raceId'] == _raceid]['statusId']:
        if iter_status not in finished_status:
            dnf_count += 1
        else:
            pass

    return dnf_count


# In[ ]:


df['dnf'] = df.raceId.apply(lambda x: count_dnf(results, x))


# In[ ]:


fig = plt.figure(figsize=(18,9))
_ = sns.scatterplot(x='points', y='dnf', data=df)
pf = np.polyfit(x=df.points, y=df.dnf, deg=1)
_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')
plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))
plt.show()


# In[ ]:


dnf_points_p = df[['points','dnf']].corr()
print(dnf_points_p)


# Doing a simple linear regression fit on race points vs. DNFs we can see a slight positive trend with more DNFs leadning to higher scores. A quick pearson coefficient calculation gives us a value of ~0.29 between points and DNFs.

# # Hypothesis 2: Amount of overtakings affects the score
# If a race has the drivers changing positions multiple times, we assume that they are either taking over or being over taken by other drives. Let's check and see if this affects the race score.

# In[ ]:


fig = plt.figure(figsize=(18,18))
plt.subplot(2,1,1)
_ = sns.lineplot(x='lap', y='position', hue='driverId', data=lapTimes[lapTimes['raceId'] == 973], palette="ch:2.5,.25")
plt.title('Driver standings during Barcelona 2017 Grand Prix')
plt.subplot(2,1,2)
_ = sns.lineplot(x='lap', y='position', hue='driverId', data=lapTimes[lapTimes['raceId'] == 864])
plt.title('Driver standings druing Barcelona 2012 Grand Prix')


# The two plots above attempt to plot the driver standings throuough the laps. Here we can see how the position of one driver changes throughout the race. Comparing the two races we see that the second race had an interesting event around lap 10 - 11 when a lot of drivers changed their position (except the ones who started in top 3).

# In[ ]:


def count_overtakings(laptimes, raceid):
    # Number of overtakings
    # The theory here is that when one driver changes his or her position between two adjacent laps, then an overtaking
    # has occurred. Counting the number of occurences this way, and then divide by 2 will give us the number of
    # overtakings since 1 overtaking includes one driver advancing one position, while the other loses one.

    competing_drivers = []
    for driver in laptimes[laptimes.raceId == raceid].driverId:
        if driver not in competing_drivers:
            competing_drivers.append(driver)

    previous_position = 0
    overtakings = 0
    for driver in competing_drivers:
        for lapPosition in laptimes[(laptimes.raceId == raceid) & (laptimes.driverId == driver)].position:
            if lapPosition != previous_position:
                previous_position = lapPosition
                overtakings += 1

    return int(overtakings/2)


# In[ ]:


df['overtakings'] = df.raceId.apply(lambda x: count_overtakings(lapTimes, x))


# In[ ]:


fig = plt.figure(figsize=(18,9))
_ = sns.scatterplot(x='points', y='overtakings', data=df)
pf = np.polyfit(x=df.points, y=df.overtakings, deg=1)
_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')
plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))


# In[ ]:


dnf_overtakings_p = df[['points','overtakings']].corr()
print(dnf_overtakings_p)


# The number of overtakings seems to signify a higher race score by the viewers!

# # Hypothesis 3: The race evolution for the top 5 drivers affects the score
# I don't know about you, but I have always remebered Formula 1 as a sport where the top 5 drivers battle it out for the podium finishes, while the rest struggle to keep up. Let's see how the evolution of the five drivers who end up in 1st to 5th place affect the score.

# In[ ]:


def get_top_5_battle(raceid, results, laptimes):
    # Focus in a race is usually on the drivers in the top, so a measurement of how their "battle" is taking shape
    # throughout the race could be interesting to measure.
    # We will approach this by looking at the variance in positions for the drivers who end up in top 5
    f_top_5 = results[(results.raceId == raceid) & (results.position < 6)].sort_values(['position'], ascending=True)
    f_top_5_var = []

    for f_driver in f_top_5.driverId:
        f_t5_var = np.var(laptimes[(laptimes.driverId == f_driver) & (laptimes.raceId == raceid)].position)
        f_top_5_var.append(f_t5_var)

    f_top5score = 0
    for f_itervar in f_top_5_var:
        f_top5score = f_top5score + f_itervar

    return f_top5score


# In[ ]:


df['top5_battle'] = df.raceId.apply(lambda x: get_top_5_battle(x, results, lapTimes))


# In[ ]:


fig = plt.figure(figsize=(18,9))
_ = sns.scatterplot(x='points', y='top5_battle', data=df)
pf = np.polyfit(x=df.points, y=df.top5_battle, deg=1)
_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')
plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))


# In[ ]:


dnf_top5_p = df[['points','top5_battle']].corr()
print(dnf_top5_p)


# More motion in the top 5 seem to be favored by the viewers!

# # Hypothesis 4: The drivers' rank and position affect the score
# As with hypothesis 3, the top 5 seem to always be the same - so does this affect the race score among the viewers?

# In[ ]:


def get_rank_vs_position(raceid, results):
    # How are the drivers rank affecting the overall satisfaction score of a race?
    # We want to test and see if the rank of the drivers in top 5 affects how good a race is, in layman's terms:
    # If a low-ranked driver finished top 5, is it more worth than if a top ranked driver wins the rays?
    f_top_5 = results[(results.raceId == raceid) & (results.position < 6)].sort_values(['position'], ascending=True)

    rvp_score = 0
    for position, rank in zip(f_top_5['position'], f_top_5['rank']):
        rvp_score += abs(position - rank)

    return rvp_score


# In[ ]:


df['rvp'] = df.raceId.apply(lambda x: get_rank_vs_position(x, results))


# In[ ]:


fig = plt.figure(figsize=(18,9))
_ = sns.scatterplot(x='points', y='rvp', data=df)
pf = np.polyfit(x=df.points, y=df.rvp, deg=1)
_ = plt.plot(np.linspace(4, 10, len(df)), pf[0]*np.linspace(4, 10, len(df)) + pf[1], 'r')
plt.title('y = kx + m where k=' + str(round(pf[0],1)) + " and m=" + str(round(pf[1],1)))


# In[ ]:


dnf_rvp_p = df[['points','rvp']].corr()
print(dnf_rvp_p)


# A positive correlation, but not as convincing as the previous hypotheses.

# # Summary
# * Hypothesis 1: Amount of DNFs affects the score. **Pearson value = 0.29**
# * Hypothesis 2: Amount of overtakings affects the score. **Pearson value = 0.36**
# * Hypothesis 3: The race evolution for the top 5 drivers affects the score. **Pearson value = 0.39**
# * Hypothesis 4: The drivers' rank and position affect the score. **Pearson value = 0.21**
# 
# So we seem to have found four features that affect the race (is it just correlation or also causation? Read "The book of why" to find out!)

# # Extra: Machine Learning
# Given the four features, can we predict the score of a race?

# In[ ]:


from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE
import optuna # Library we use for hyperparameter tuning


# In[ ]:


X = df[['raceId','dnf','overtakings','top5_battle','rvp']]
y = df['points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# We will use an xgboost regressor for this example, and we will train two different models: one with hyperparameter tuning and one without.
# 
# Let's start with the tuned one:

# In[ ]:


# Hyperparameter tuning
def objective(trial):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    op_params = {
        'gamma': trial.suggest_uniform('gamma', 0.1, 1),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.1, 0.9),
        'max_depth': trial.suggest_int('max_depth', 1, 5),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000),
        'col_sample_by_tree': trial.suggest_uniform('col_sample_by_tree', 0.1, 0.5),
        'booster': 'gbtree'
    }
    
    op_model = xgb.train(op_params, dtrain)
    op_preds = op_model.predict(dtest)
    
    return MSE(op_preds, y_test)

study = optuna.create_study()
study.optimize(objective, n_trials=30)


# In[ ]:


params = study.best_params
params['booster'] = 'gbtree'
dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train(params, dtrain)


# Then, we train an "untuned" model as well:

# In[ ]:


untuned_model = xgb.XGBRegressor(seed=42)
untuned_model.fit(X_train, y_train)
untuned_predictions = untuned_model.predict(X_test)


# In[ ]:


dtest = xgb.DMatrix(X_test, label=y_test)
predictions = model.predict(dtest)


# Comparing the two models, as well as a naive model which always gives the mean score of racefans.net ratings.

# In[ ]:


acc = MSE(predictions, y_test)
print("Tuned model MSE score: " + str(acc))
avg_sc = np.ones((len(predictions), 1))*df.points.mean()
baseline = MSE(avg_sc, y_test)
print("Baseline score: " + str(baseline))
untuned_acc = MSE(untuned_predictions, y_test)
print("Untuned model score: " + str(untuned_acc))


# We see that the MSE of the tuned model is significantly better than both the untuned model as well as the naive "mean points model". Let's take the tuned model's predictions and compare to the real values in the test set.

# In[ ]:


pred_compare = pd.DataFrame({
    'real': y_test.values,
    'predictions': predictions
})
pred_compare = pred_compare.sort_values('real')


# In[ ]:


x_sup = np.linspace(0,len(predictions), len(predictions))
fig = plt.figure(figsize=(18,9))
_ = plt.plot(x_sup, pred_compare['real'], 'bo', x_sup, pred_compare['predictions'], 'rx')
plt.ylim(4,10)
plt.legend(['Tuned Model Predictions', 'Real Scores'])


# Well, it is not perfect, and it seems to be especially hard to awared the correct points for really good races (>8.5 points) and the really bad ones (<6.0 points).
# 
# In conclusion, it seems like we can do a somewhat good job at prediction the race score based on DNFs, overtakings, top5 battle and driver rank, but it is obviously not enough and we are most likely missing important features in our analysis. On top of my head I can think of weather, track and how far in we are in the season. Maybe that is something someone else can analyze :) 
# 
# **//Christian**

# In[ ]:




